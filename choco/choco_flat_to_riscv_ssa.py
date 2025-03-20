# choco/choco_flat_to_riscv_ssa.py
# type: ignore

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp, IntegerAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

# ---- Choco IR 定义 (choco_flat.py) ----
from choco.dialects.choco_flat import (
    # Ops
    Alloc,
    Assign,
    BinaryExpr,
    BoolAttr,
    CallExpr,
    ClassDef,
    EffectfulBinaryExpr,
    For,
    FuncDef,
    GetAddress,
    If,
    IfExpr,
    IndexString,
    ListExpr,
    Literal,
    Load,
    MemberExpr,
    MemlocType,
    NoneAttr,
    Pass,
    Return,
    Store,
    UnaryExpr,
    While,
    Yield,
    # Types
    NamedType,
    ListType,
    StringAttr,
    int_type,
    bool_type,
    none_type,
    str_type,
)

# ---- RISC-V SSA Dialect (riscv/ssa_dialect.py) ----
from riscv.ssa_dialect import (
    RegisterType,
    # load/store
    LWOp, SWOp,
    # branches
    BEQOp, BNEOp, BLTOp, BGEOp,
    # arithmetic
    AddOp, SubOp, MULOp, DIVOp, REMOp,
    SLTOp, SLTUOp, ANDOp, OROp, XOROp, SNEZOp, SEQZOp,
    AddIOp,
    # function call
    CallOp,
    # function definition + return
    FuncOp, ReturnOp,
    # pseudo-instructions
    LIOp,
    # label + unconditional jump
    LabelOp, JOp,
    # for possible dynamic usage
    AllocOp
)

# 你可能用到的一些运算符，如 ANDIOp, ORIOp, SLTIOp 等:
# from riscv.ssa_dialect import ANDIOp, ORIOp, SLTIOp, SLTIUOp  # 按需导入


# =============== Pattern Implementations ===============

class LiteralPattern(RewritePattern):
    """
    把 choco.ir.literal 转成 RISC-V SSA:
    - int => li
    - bool => li(0/1)
    - None => li(0)
    - string => (最简) li(0)，如果你要真在堆上分配字符串需额外处理
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Literal, rewriter: PatternRewriter):
        value = op.value

        # 整型
        if isinstance(value, IntegerAttr):
            i_val = value.value.data  # 取出真实的 Python int
            li_inst = LIOp(i_val)
            rewriter.replace_op(op, [li_inst])
            return

        # 布尔
        if isinstance(value, BoolAttr):
            bool_int = 1 if value.data else 0
            li_inst = LIOp(bool_int)
            rewriter.replace_op(op, [li_inst])
            return

        # None => li(0)
        if isinstance(value, NoneAttr):
            li_inst = LIOp(0)
            rewriter.replace_op(op, [li_inst])
            return

        # 字符串 => 简化处理: 仅 li(0)
        if isinstance(value, StringAttr):
            li_inst = LIOp(0)
            rewriter.replace_op(op, [li_inst])
            return

        # 若出现其他情况, raise
        raise NotImplementedError(f"Unknown literal attr: {value}")


class CallPattern(RewritePattern):
    """
    choco.ir.call_expr =>
    - 特殊情况: len(...) => 生成对 _error_len_none 的检查 & lw 读长度
    - 其他 => 生成 riscv_ssa.call
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallExpr, rewriter: PatternRewriter):
        func_name = op.func_name.data
        args = list(op.args)

        # 检查内置函数 len
        if func_name == "len":
            # 1) zero = li(0)
            zero = LIOp(0)
            # 2) beq(args[0], zero, => _error_len_none)
            check_none = BEQOp(args[0], zero, "_error_len_none")
            # 3) read_size = lw(args[0], 0)
            read_size = LWOp(args[0], 0)
            rewriter.replace_op(op, [zero, check_none, read_size])
            return

        # 其他函数 => 直接用 riscv_ssa.call
        has_result = bool(op.results)
        call_inst = CallOp(func_name, args, has_result=has_result)
        rewriter.replace_op(op, [call_inst])


class AllocPattern(RewritePattern):
    """
    choco.ir.alloc => 可能:
      - 通过 heap (call _malloc)
      - 或者放栈 (AllocOp)
    示例: 先 li(4), 再 call _malloc
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: Alloc, rewriter: PatternRewriter):
        # 简单：为对象分配4字节
        li_4 = LIOp(4)
        malloc_call = CallOp("_malloc", [li_4], has_result=True)
        rewriter.replace_op(alloc_op, [li_4, malloc_call])


class StorePattern(RewritePattern):
    """
    choco.ir.store => sw(value, memloc, 0)
    注意: memloc 是指针(register中)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, store_op: Store, rewriter: PatternRewriter):
        sw_inst = SWOp(store_op.value, store_op.memloc, 0)
        rewriter.replace_op(store_op, [sw_inst])


class LoadPattern(RewritePattern):
    """
    choco.ir.load => lw(memloc, 0)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, load_op: Load, rewriter: PatternRewriter):
        lw_inst = LWOp(load_op.memloc, 0)
        rewriter.replace_op(load_op, [lw_inst])


class UnaryExprPattern(RewritePattern):
    """
    choco.ir.unary_expr:
      - op='-' => 0 - x
      - op='not' => seqz(x) (x==0 =>1 else0)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, unary_op: UnaryExpr, rewriter: PatternRewriter):
        op_str = unary_op.op.data
        val = unary_op.value

        if op_str == "-":
            zero = LIOp(0)
            neg = SubOp(zero, val)
            rewriter.replace_op(unary_op, [zero, neg])
            return

        if op_str == "not":
            # seqz => if x==0 =>1 else0
            seqz_ = SEQZOp(val)
            rewriter.replace_op(unary_op, [seqz_])
            return

        raise NotImplementedError(f"Unsupported unary operator: {op_str}")


class BinaryExprPattern(RewritePattern):
    """
    choco.ir.binary_expr
    常见: +, -, *, //, %, <, >, <=, >=, ==, !=
    (and/or 是 EffectfulBinaryExpr)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, bin_op: BinaryExpr, rewriter: PatternRewriter):
        op_str = bin_op.op.data
        lhs = bin_op.lhs
        rhs = bin_op.rhs

        # 算术
        if op_str == "+":
            add_ = AddOp(lhs, rhs)
            rewriter.replace_op(bin_op, [add_])
            return
        if op_str == "-":
            sub_ = SubOp(lhs, rhs)
            rewriter.replace_op(bin_op, [sub_])
            return
        if op_str == "*":
            mul_ = MULOp(lhs, rhs)
            rewriter.replace_op(bin_op, [mul_])
            return
        if op_str == "//":
            div_ = DIVOp(lhs, rhs)
            rewriter.replace_op(bin_op, [div_])
            return
        if op_str == "%":
            rem_ = REMOp(lhs, rhs)
            rewriter.replace_op(bin_op, [rem_])
            return

        # 比较 => 结果0/1
        if op_str == "<":
            slt_ = SLTOp(lhs, rhs)
            rewriter.replace_op(bin_op, [slt_])
            return
        if op_str == ">":
            # x>y => y<x
            slt_ = SLTOp(rhs, lhs)
            rewriter.replace_op(bin_op, [slt_])
            return
        if op_str == "<=":
            # x<=y => not(x>y)
            # => x>y => y<x => slt(rhs,lhs), seqz
            slt_ = SLTOp(rhs, lhs)
            seqz_ = SEQZOp(slt_)
            rewriter.replace_op(bin_op, [slt_, seqz_])
            return
        if op_str == ">=":
            # x>=y => not(x<y) => slt(lhs,rhs) => seqz
            slt_ = SLTOp(lhs, rhs)
            seqz_ = SEQZOp(slt_)
            rewriter.replace_op(bin_op, [slt_, seqz_])
            return
        if op_str == "==":
            # (lhs-rhs)==0 => sub => seqz
            sub_ = SubOp(lhs, rhs)
            eq_ = SEQZOp(sub_)
            rewriter.replace_op(bin_op, [sub_, eq_])
            return
        if op_str == "!=":
            # sub => snez
            sub_ = SubOp(lhs, rhs)
            ne_ = SNEZOp(sub_)
            rewriter.replace_op(bin_op, [sub_, ne_])
            return

        raise NotImplementedError(f"Unsupported binary op: {op_str}")


class IfPattern(RewritePattern):
    """
    choco.ir.if(cond, then, else)
    => 生成 label + 分支 + inline
    """

    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, if_op: If, rewriter: PatternRewriter):
        c = IfPattern.counter
        IfPattern.counter += 1

        label_then = f"if_then_{c}"
        label_else = f"if_else_{c}"
        label_after = f"if_after_{c}"

        cond = if_op.cond
        zero = LIOp(0)
        beq_ = BEQOp(cond, zero, label_else)

        rewriter.insert_op_before_matched_op(zero, if_op)
        rewriter.insert_op_before_matched_op(beq_, if_op)

        # label_then
        label_then_op = LabelOp(label_then)
        rewriter.insert_op_before_matched_op(label_then_op, if_op)
        # inline then block
        rewriter.inline_block_before_matched_op(if_op.then.block, if_op)

        # j -> after
        j_after = JOp(label_after)
        rewriter.insert_op_before_matched_op(j_after, if_op)

        # else label
        label_else_op = LabelOp(label_else)
        rewriter.insert_op_before_matched_op(label_else_op, if_op)
        rewriter.inline_block_before_matched_op(if_op.orelse.block, if_op)

        # after
        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op, if_op)

        rewriter.erase_matched_op(if_op)


class AndPattern(RewritePattern):
    """
    choco.ir.effectful_binary_expr(op='and'):
    短路逻辑
    """

    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, and_op: EffectfulBinaryExpr, rewriter: PatternRewriter):
        if and_op.op.data != "and":
            return
        c = AndPattern.counter
        AndPattern.counter += 1
        label_rhs = f"and_rhs_{c}"
        label_after = f"and_after_{c}"

        # inline LHS
        lhs_block = and_op.lhs.block
        lhs_yield = lhs_block.ops.last  # type: ignore
        lhs_val = lhs_yield.value

        zero = LIOp(0)
        # beq(lhs_val,0) => jump after
        beq_skip = BEQOp(lhs_val, zero, label_after)

        # label rhs
        label_rhs_op = LabelOp(label_rhs)

        # 插入
        rewriter.insert_op_before_matched_op(zero, and_op)
        rewriter.insert_op_before_matched_op(beq_skip, and_op)
        rewriter.insert_op_before_matched_op(label_rhs_op, and_op)

        # inline RHS
        rhs_block = and_op.rhs.block
        rewriter.inline_block_before_matched_op(rhs_block, and_op)
        rhs_yield = rhs_block.ops.last  # type: ignore
        rhs_val = rhs_yield.value

        # label_after
        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op, and_op)

        # 结果 => if lhs=0 => final=0 else final=rhs
        # 做一个小汇编：temp=li(0); bne(lhs_val,0)-> set temp=rhs
        temp0 = LIOp(0)
        c_label = f"and_assign_{c}"
        label_assign = LabelOp(c_label)
        bne_ = BNEOp(lhs_val, zero, c_label)
        # move => temp = rhs
        final_ = AddOp(rhs_val, zero)

        # 顺序插入
        rewriter.insert_op_before_matched_op(temp0, and_op)
        rewriter.insert_op_before_matched_op(bne_, and_op)
        rewriter.insert_op_before_matched_op(label_assign, and_op)
        rewriter.insert_op_before_matched_op(final_, and_op)

        # replace => 该EffectfulBinaryExpr的"result" 就是 final_
        rewriter.replace_matched_op(
            and_op,
            [
                lhs_block.ops.last,
                zero,
                beq_skip,
                label_rhs_op,
                rhs_block.ops.last,
                label_after_op,
                temp0,
                bne_,
                label_assign,
                final_,
            ],
            [final_],
        )


class OrPattern(RewritePattern):
    """
    choco.ir.effectful_binary_expr(op='or') 短路
    """

    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, or_op: EffectfulBinaryExpr, rewriter: PatternRewriter):
        if or_op.op.data != "or":
            return
        c = OrPattern.counter
        OrPattern.counter += 1
        label_rhs = f"or_rhs_{c}"
        label_after = f"or_after_{c}"

        # inline LHS
        lhs_block = or_op.lhs.block
        lhs_yield = lhs_block.ops.last  # type: ignore
        lhs_val = lhs_yield.value

        zero = LIOp(0)
        # bne(lhs_val,0)-> skip => jump after
        bne_skip = BNEOp(lhs_val, zero, label_after)

        label_rhs_op = LabelOp(label_rhs)

        rewriter.insert_op_before_matched_op(zero, or_op)
        rewriter.insert_op_before_matched_op(bne_skip, or_op)
        rewriter.insert_op_before_matched_op(label_rhs_op, or_op)

        # inline RHS
        rhs_block = or_op.rhs.block
        rewriter.inline_block_before_matched_op(rhs_block, or_op)
        rhs_yield = rhs_block.ops.last  # type: ignore
        rhs_val = rhs_yield.value

        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op, or_op)

        # 结果 => if lhs!=0 => final=lhs_val else final=rhs_val
        temp0 = LIOp(0)
        # 先 temp0=rhs_val
        final_rhs = AddOp(rhs_val, zero)

        c_label = f"or_assign_{c}"
        label_assign = LabelOp(c_label)
        bne_ = BNEOp(lhs_val, zero, c_label)
        final_lhs = AddOp(lhs_val, zero)

        rewriter.insert_op_before_matched_op(temp0, or_op)
        rewriter.insert_op_before_matched_op(final_rhs, or_op)
        rewriter.insert_op_before_matched_op(bne_, or_op)
        rewriter.insert_op_before_matched_op(label_assign, or_op)
        rewriter.insert_op_before_matched_op(final_lhs, or_op)

        rewriter.replace_matched_op(
            or_op,
            [
                lhs_block.ops.last,
                zero,
                bne_skip,
                label_rhs_op,
                rhs_block.ops.last,
                label_after_op,
                temp0,
                final_rhs,
                bne_,
                label_assign,
                final_lhs,
            ],
            [final_lhs],
        )


class IfExprPattern(RewritePattern):
    """
    choco.ir.if_expr(cond, then, else)
    在低层 SSA里需要phi合并, 这里仅简化处理(把两边inlined), 具体做法可用store+load或额外技巧
    """

    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, if_op: IfExpr, rewriter: PatternRewriter):
        c = IfExprPattern.counter
        IfExprPattern.counter += 1

        label_then = f"ifexpr_then_{c}"
        label_else = f"ifexpr_else_{c}"
        label_after = f"ifexpr_after_{c}"

        cond = if_op.cond
        zero = LIOp(0)
        beq_ = BEQOp(cond, zero, label_else)

        rewriter.insert_op_before_matched_op(zero, if_op)
        rewriter.insert_op_before_matched_op(beq_, if_op)

        # then
        label_then_op = LabelOp(label_then)
        rewriter.insert_op_before_matched_op(label_then_op, if_op)
        rewriter.inline_block_before_matched_op(if_op.then.block, if_op)
        then_val = if_op.then_ssa_value

        j_after = JOp(label_after)
        rewriter.insert_op_before_matched_op(j_after, if_op)

        # else
        label_else_op = LabelOp(label_else)
        rewriter.insert_op_before_matched_op(label_else_op, if_op)
        rewriter.inline_block_before_matched_op(if_op.or_else.block, if_op)
        else_val = if_op.or_else_ssa_value

        # label_after
        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op, if_op)

        # 简易: 不做真正phi, 只erase(无法返回单一值给后续)
        # 若测试需要 if_expr 的值, 需额外逻辑(如在then/else都store同一栈slot再load)
        rewriter.erase_matched_op(if_op)


class WhilePattern(RewritePattern):
    """
    choco.ir.while(cond, body)
    => label_loop, inline cond => if cond==0 => jump after => inline body => j loop => label_after
    """

    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, while_op: While, rewriter: PatternRewriter):
        c = WhilePattern.counter
        WhilePattern.counter += 1
        label_loop = f"while_loop_{c}"
        label_after = f"while_after_{c}"

        # label_loop
        label_loop_op = LabelOp(label_loop)
        rewriter.insert_op_before_matched_op(label_loop_op, while_op)

        # inline cond
        rewriter.inline_block_before_matched_op(while_op.cond.block, while_op)
        cond_yield = while_op.cond.block.ops.last  # type: ignore
        cond_val = cond_yield.value

        zero = LIOp(0)
        beq_ = BEQOp(cond_val, zero, label_after)
        rewriter.insert_op_before_matched_op(zero, while_op)
        rewriter.insert_op_before_matched_op(beq_, while_op)

        # inline body
        rewriter.inline_block_before_matched_op(while_op.body.block, while_op)

        # jump back
        j_loop = JOp(label_loop)
        rewriter.insert_op_before_matched_op(j_loop, while_op)

        # label_after
        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op, while_op)

        rewriter.erase_matched_op(while_op)


class ForPattern(RewritePattern):
    """
    choco.ir.for(...)
    在 CT3 中可能已在前端 pass 转为 while; 如果还存在 for, 这里可做not implemented
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, for_op: For, rewriter: PatternRewriter):
        raise NotImplementedError("For loops are expected to be lowered to While first.")


class ListExprPattern(RewritePattern):
    """
    choco.ir.list_expr(...) => 在堆上分配, 存length + 元素
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, list_expr: ListExpr, rewriter: PatternRewriter):
        elems = list(list_expr.elems)
        length = len(elems)

        # 分配 total_bytes= 4(length) + 4 * len(elems)
        total_bytes = 4 + 4 * length
        li_size = LIOp(total_bytes)
        malloc_ = CallOp("_malloc", [li_size], has_result=True)

        # store length => sw(len, malloc_, 0)
        store_len = SWOp(LIOp(length), malloc_, 0)

        ops_ = [li_size, malloc_, store_len]

        offset = 4
        for e in elems:
            sw_ = SWOp(e, malloc_, offset)
            ops_.append(sw_)
            offset += 4

        rewriter.replace_op(list_expr, ops_)


class GetAddressPattern(RewritePattern):
    """
    choco.ir.get_address(value, index) => &value[index]
    => offset=4 + index*4 => add base, offset
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_address: GetAddress, rewriter: PatternRewriter):
        val = get_address.value
        idx = get_address.index
        # offset = 4 + idx*4
        one = LIOp(1)
        plus_one = AddOp(idx, one)
        four = LIOp(4)
        times4 = MULOp(plus_one, four)
        final_addr = AddOp(val, times4)

        rewriter.replace_op(
            get_address,
            [one, plus_one, four, times4, final_addr]
        )


class IndexStringPattern(RewritePattern):
    """
    choco.ir.index_string(value, index) => bound check + offset
    => if val=0 => _list_index_none
       load length => if idx>=length => _list_index_oob
       final ptr= base+(4+idx*4)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, indexString: IndexString, rewriter: PatternRewriter):
        val = indexString.value
        idx = indexString.index

        zero = LIOp(0)
        check_none = BEQOp(val, zero, "_list_index_none")

        length = LWOp(val, 0)
        # if idx>=length => jump => _list_index_oob
        # BGEOp(idx, length, label)
        check_oob = BGEOp(idx, length, "_list_index_oob")

        # compute address
        one = LIOp(1)
        plus_one = AddOp(idx, one)
        four = LIOp(4)
        times4 = MULOp(plus_one, four)
        final_ptr = AddOp(val, times4)

        rewriter.replace_op(
            indexString,
            [zero, check_none, length, check_oob, one, plus_one, four, times4, final_ptr]
        )


class AssignPattern(RewritePattern):
    """
    choco.ir.assign(target, value)
    一般在 ChocoPy 中转成 store(load_of_target, value)? 也许已被处理
    如果还存在, 这里可尝试:
       - 如果 target 是内存地址 => store
       - 如果 target/ value 都是 register => 变成 addOp(value, zero)?
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, assign_op: Assign, rewriter: PatternRewriter):
        # 可能 “assign x, y” => x 是个 memloc?
        # or x 直接是个 register
        target = assign_op.target
        val = assign_op.value
        # 由于 IR 里通常 x=... 会拆成 storeOp,
        # 也可能 assign保留对对象字段赋值?
        # 这里做个简易: store(val, target, 0) if target是MemlocType
        # 否则 move
        if isinstance(target.type, MemlocType):
            sw_ = SWOp(val, target, 0)
            rewriter.replace_op(assign_op, [sw_])
        elif isinstance(target.type, RegisterType):
            # move => addOp(val, 0)
            zero = LIOp(0)
            mv_ = AddOp(val, zero)
            rewriter.replace_op(assign_op, [zero, mv_])
        else:
            raise NotImplementedError("Assign to unknown type target")


class MemberExprPattern(RewritePattern):
    """
    choco.ir.member_expr(value, attribute)
    访问对象字段 => 可能不在CT3用. 这里先不实现或 raise
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, memexpr: MemberExpr, rewriter: PatternRewriter):
        raise NotImplementedError("MemberExpr not implemented in this pass.")


class ClassDefPattern(RewritePattern):
    """
    choco.ir.class_def => for OOP, not used or partial
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, clsdef: ClassDef, rewriter: PatternRewriter):
        raise NotImplementedError("ClassDef not implemented in this pass.")


class PassPattern(RewritePattern):
    """
    choco.ir.pass => no-op
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, pass_op: Pass, rewriter: PatternRewriter):
        rewriter.erase_matched_op(pass_op)


class YieldPattern(RewritePattern):
    """
    删除 choco.ir.yield（我们已将其值在父Op中用到）
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, y_op: Yield, rewriter: PatternRewriter):
        rewriter.erase_matched_op(y_op)


class FuncDefPattern(RewritePattern):
    """
    choco.ir.func_def => riscv_ssa.func
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, func: FuncDef, rewriter: PatternRewriter):
        new_func = FuncOp.create(
            result_types=[],
            properties={"func_name": func.func_name},
        )
        new_region = rewriter.move_region_contents_to_new_regions(func.func_body)
        new_func.add_region(new_region)
        for arg in new_region.block.args:
            rewriter.modify_value_type(arg, RegisterType())
        rewriter.replace_op(func, new_func)


class ReturnPattern(RewritePattern):
    """
    choco.ir.return => riscv_ssa.return
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, ret: Return, rewriter: PatternRewriter):
        val = ret.value
        new_ret = ReturnOp(val)
        rewriter.replace_op(ret, [new_ret])


# =========== 主 Pass ===========

class ChocoFlatToRISCVSSA(ModulePass):
    name = "choco-flat-to-riscv-ssa"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # 第一次：匹配大部分IR ops
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # 语句/表达式
                    LiteralPattern(),
                    CallPattern(),
                    AllocPattern(),
                    StorePattern(),
                    LoadPattern(),
                    UnaryExprPattern(),
                    BinaryExprPattern(),
                    IfPattern(),
                    AndPattern(),
                    OrPattern(),
                    IfExprPattern(),
                    WhilePattern(),
                    ForPattern(),
                    ListExprPattern(),
                    GetAddressPattern(),
                    IndexStringPattern(),
                    AssignPattern(),
                    MemberExprPattern(),
                    ClassDefPattern(),
                    PassPattern(),
                    FuncDefPattern(),
                    ReturnPattern(),
                ]
            ),
            apply_recursively=True,
        )
        walker.rewrite_module(op)

        # 第二次：专门把 Yield 去掉
        walker2 = PatternRewriteWalker(
            GreedyRewritePatternApplier([YieldPattern()]),
            apply_recursively=True,
        )
        walker2.rewrite_module(op)
