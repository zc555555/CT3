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


def _get_or_insert_block_result(block, rewriter: PatternRewriter, default_val: int = 1):
    """
    辅助函数: 获取 block 最后一条指令的 .value (若存在), 否则插入 li(default_val) 返回.
    用于应对空 block 或最后指令不产生值时的情况, 避免 NoneType 错误.
    """
    last_op = block.ops.last
    if not last_op:
        # block 为空，插入 li(default_val)
        fallback = LIOp(default_val)
        # 注意: 我们无法插入到 block 本身(因为 inline 时机会已过),
        # 暂时只返回 fallback, 由调用者将其插到正确的位置
        return fallback, True
    # 如果最后指令有 'value' 属性则用它
    val = getattr(last_op, 'value', None)
    if val is not None:
        return val, False
    # 否则插入 fallback
    fallback = LIOp(default_val)
    return fallback, True


# =============== Pattern Implementations ===============

class LiteralPattern(RewritePattern):
    """
    把 choco.ir.literal 转成 RISC-V SSA:
    - int => li
    - bool => li(0/1)
    - None => li(0)
    - string => (最简) li(0)  (可扩展成堆上分配)
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Literal, rewriter: PatternRewriter):
        value = op.value

        # 整型
        if isinstance(value, IntegerAttr):
            i_val = value.value.data  # Python int
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

        raise NotImplementedError(f"Unknown literal attr: {value}")


class CallPattern(RewritePattern):
    """
    choco.ir.call_expr => riscv_ssa.call
    特殊: len(...) => beq => lw
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallExpr, rewriter: PatternRewriter):
        func_name = op.func_name.data
        args = list(op.args)

        # check len
        if func_name == "len":
            zero = LIOp(0)
            check_none = BEQOp(args[0], zero, "_error_len_none")
            read_size = LWOp(args[0], 0)
            rewriter.replace_op(op, [zero, check_none, read_size])
            return

        # normal call
        has_result = bool(op.results)
        call_inst = CallOp(func_name, args, has_result=has_result)
        rewriter.replace_op(op, [call_inst])


class AllocPattern(RewritePattern):
    """
    choco.ir.alloc => (示例) li(4), call _malloc
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: Alloc, rewriter: PatternRewriter):
        li_4 = LIOp(4)
        malloc_call = CallOp("_malloc", [li_4], has_result=True)
        rewriter.replace_op(alloc_op, [li_4, malloc_call])


class StorePattern(RewritePattern):
    """
    store => sw(value, memloc, 0)
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, store_op: Store, rewriter: PatternRewriter):
        sw_inst = SWOp(store_op.value, store_op.memloc, 0)
        rewriter.replace_op(store_op, [sw_inst])


class LoadPattern(RewritePattern):
    """
    load => lw(memloc, 0)
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, load_op: Load, rewriter: PatternRewriter):
        lw_inst = LWOp(load_op.memloc, 0)
        rewriter.replace_op(load_op, [lw_inst])


class UnaryExprPattern(RewritePattern):
    """
    unary_expr => -x or not x
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
            # seqz => (x==0 =>1, else 0)
            seqz_ = SEQZOp(val)
            rewriter.replace_op(unary_op, [seqz_])
            return

        raise NotImplementedError(f"Unsupported unary operator: {op_str}")


class BinaryExprPattern(RewritePattern):
    """
    binary_expr => +, -, *, //, %, <, >, <=, >=, ==, !=, is, is not, and, or
    (无副作用场景 => 只计算布尔结果0/1)
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

        # 比较 => 0/1
        if op_str == "<":
            slt_ = SLTOp(lhs, rhs)
            rewriter.replace_op(bin_op, [slt_])
            return
        if op_str == ">":
            # x>y => y<x => slt(rhs, lhs)
            slt_ = SLTOp(rhs, lhs)
            rewriter.replace_op(bin_op, [slt_])
            return
        if op_str == "<=":
            # x<=y => not(x>y)
            slt_ = SLTOp(rhs, lhs)
            seqz_ = SEQZOp(slt_)
            rewriter.replace_op(bin_op, [slt_, seqz_])
            return
        if op_str == ">=":
            # x>=y => not(x<y)
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

        # Python 中的 "is" / "is not"
        if op_str == "is":
            sub_ = SubOp(lhs, rhs)
            eq_ = SEQZOp(sub_)
            rewriter.replace_op(bin_op, [sub_, eq_])
            return
        if op_str == "is not":
            sub_ = SubOp(lhs, rhs)
            ne_ = SNEZOp(sub_)
            rewriter.replace_op(bin_op, [sub_, ne_])
            return

        # 处理无副作用的 and / or => 返回布尔值0/1
        if op_str == "and":
            snez_lhs = SNEZOp(lhs)
            snez_rhs = SNEZOp(rhs)
            and_ = ANDOp(snez_lhs, snez_rhs)
            rewriter.replace_op(bin_op, [snez_lhs, snez_rhs, and_])
            return
        if op_str == "or":
            snez_lhs = SNEZOp(lhs)
            snez_rhs = SNEZOp(rhs)
            or_ = OROp(snez_lhs, snez_rhs)
            rewriter.replace_op(bin_op, [snez_lhs, snez_rhs, or_])
            return

        raise NotImplementedError(f"Unsupported binary op: {op_str}")


class IfPattern(RewritePattern):
    """
    if(cond, then, else):
    => cond==0 -> else; otherwise then; end
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

        rewriter.insert_op_before_matched_op(zero)
        rewriter.insert_op_before_matched_op(beq_)

        label_then_op = LabelOp(label_then)
        rewriter.insert_op_before_matched_op(label_then_op)
        rewriter.inline_block_before_matched_op(if_op.then.block)

        j_after = JOp(label_after)
        rewriter.insert_op_before_matched_op(j_after)

        label_else_op = LabelOp(label_else)
        rewriter.insert_op_before_matched_op(label_else_op)
        rewriter.inline_block_before_matched_op(if_op.orelse.block)

        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op)

        rewriter.erase_matched_op(if_op)


class AndPattern(RewritePattern):
    """
    effectful_binary_expr(op='and') => 短路
    用于有副作用子表达式的短路场景
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
        label_assign = f"and_assign_{c}"

        # inline LHS block 之前, 先判断 LHS block 最后是否产生值
        lhs_block = and_op.lhs.block
        lhs_val, need_insert_lhs = _get_or_insert_block_result(lhs_block, rewriter, default_val=1)

        # 现在插入 LHS block
        rewriter.insert_op_before_matched_op(lhs_val) if need_insert_lhs else None
        rewriter.inline_block_before_matched_op(lhs_block)

        zero = LIOp(0)
        beq_skip = BEQOp(lhs_val, zero, label_after)
        rewriter.insert_op_before_matched_op(zero)
        rewriter.insert_op_before_matched_op(beq_skip)

        label_rhs_op = LabelOp(label_rhs)
        rewriter.insert_op_before_matched_op(label_rhs_op)

        # inline RHS block
        rhs_block = and_op.rhs.block
        rhs_val, need_insert_rhs = _get_or_insert_block_result(rhs_block, rewriter, default_val=1)
        if need_insert_rhs:
            rewriter.insert_op_before_matched_op(rhs_val)
        rewriter.inline_block_before_matched_op(rhs_block)

        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op)

        # 最终结果：若 lhs=0 则整体为lhs，否则为rhs
        temp0 = LIOp(0)
        bne_ = BNEOp(lhs_val, zero, label_assign)
        rewriter.insert_op_before_matched_op(temp0)
        rewriter.insert_op_before_matched_op(bne_)

        label_assign_op = LabelOp(label_assign)
        rewriter.insert_op_before_matched_op(label_assign_op)

        final_ = AddOp(rhs_val, temp0)  # => rhs_val + 0
        # 全部替换
        rewriter.replace_matched_op(
            and_op,
            [
                lhs_val if need_insert_lhs else (),
                zero,
                beq_skip,
                label_rhs_op,
                rhs_val if need_insert_rhs else (),
                label_after_op,
                temp0,
                bne_,
                label_assign_op,
                final_,
            ],
            [final_],
        )


class OrPattern(RewritePattern):
    """
    effectful_binary_expr(op='or') => 短路
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
        label_assign = f"or_assign_{c}"

        # inline LHS block
        lhs_block = or_op.lhs.block
        lhs_val, need_insert_lhs = _get_or_insert_block_result(lhs_block, rewriter, default_val=0)
        if need_insert_lhs:
            rewriter.insert_op_before_matched_op(lhs_val)
        rewriter.inline_block_before_matched_op(lhs_block)

        zero = LIOp(0)
        bne_skip = BNEOp(lhs_val, zero, label_after)
        rewriter.insert_op_before_matched_op(zero)
        rewriter.insert_op_before_matched_op(bne_skip)

        label_rhs_op = LabelOp(label_rhs)
        rewriter.insert_op_before_matched_op(label_rhs_op)

        # inline RHS block
        rhs_block = or_op.rhs.block
        rhs_val, need_insert_rhs = _get_or_insert_block_result(rhs_block, rewriter, default_val=1)
        if need_insert_rhs:
            rewriter.insert_op_before_matched_op(rhs_val)
        rewriter.inline_block_before_matched_op(rhs_block)

        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op)

        # unify final: 若 lhs!=0 则结果为lhs, 否则rhs
        temp0 = LIOp(0)
        final_rhs = AddOp(rhs_val, temp0)
        bne_ = BNEOp(lhs_val, zero, label_assign)
        rewriter.insert_op_before_matched_op(temp0)
        rewriter.insert_op_before_matched_op(final_rhs)
        rewriter.insert_op_before_matched_op(bne_)

        label_assign_op = LabelOp(label_assign)
        rewriter.insert_op_before_matched_op(label_assign_op)

        final_lhs = AddOp(lhs_val, temp0)
        rewriter.replace_matched_op(
            or_op,
            [
                lhs_val if need_insert_lhs else (),
                zero,
                bne_skip,
                label_rhs_op,
                rhs_val if need_insert_rhs else (),
                label_after_op,
                temp0,
                final_rhs,
                bne_,
                label_assign_op,
                final_lhs,
            ],
            [final_lhs],
        )


class IfExprPattern(RewritePattern):
    """
    if_expr(cond, then, else):
    => inline cond -> if=0 => else ; else => then
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

        rewriter.insert_op_before_matched_op(zero)
        rewriter.insert_op_before_matched_op(beq_)

        label_then_op = LabelOp(label_then)
        rewriter.insert_op_before_matched_op(label_then_op)
        rewriter.inline_block_before_matched_op(if_op.then.block)
        # then_val = if_op.then_ssa_value  # 不做实际phi; 仅执行

        j_after = JOp(label_after)
        rewriter.insert_op_before_matched_op(j_after)

        label_else_op = LabelOp(label_else)
        rewriter.insert_op_before_matched_op(label_else_op)
        rewriter.inline_block_before_matched_op(if_op.or_else.block)
        # else_val = if_op.or_else_ssa_value

        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op)

        # 不做真正的 phi
        rewriter.erase_matched_op(if_op)


class WhilePattern(RewritePattern):
    """
    while(cond, body):
    => label_loop => inline cond => beq(cond,0)->after => inline body => j(loop) => label_after
    """
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, while_op: While, rewriter: PatternRewriter):
        c = WhilePattern.counter
        WhilePattern.counter += 1
        label_loop = f"while_loop_{c}"
        label_after = f"while_after_{c}"

        label_loop_op = LabelOp(label_loop)
        rewriter.insert_op_before_matched_op(label_loop_op)

        rewriter.inline_block_before_matched_op(while_op.cond.block)
        cond_yield = while_op.cond.block.ops.last
        cond_val = cond_yield.value if cond_yield and hasattr(cond_yield, 'value') else LIOp(1)
        if cond_yield is None:
            # 如果cond block完全空, 这里插入 fallback
            rewriter.insert_op_before_matched_op(cond_val)

        zero = LIOp(0)
        beq_ = BEQOp(cond_val, zero, label_after)
        rewriter.insert_op_before_matched_op(zero)
        rewriter.insert_op_before_matched_op(beq_)

        rewriter.inline_block_before_matched_op(while_op.body.block)
        j_loop = JOp(label_loop)
        rewriter.insert_op_before_matched_op(j_loop)

        label_after_op = LabelOp(label_after)
        rewriter.insert_op_before_matched_op(label_after_op)

        rewriter.erase_matched_op(while_op)


class ForPattern(RewritePattern):
    """
    If the IR still has "for", either lower it or raise NotImplemented
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, for_op: For, rewriter: PatternRewriter):
        raise NotImplementedError("For loops are expected to be lowered to While first.")


class ListExprPattern(RewritePattern):
    """
    list_expr => heap allocate [length + elements], 返回指针
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, list_expr: ListExpr, rewriter: PatternRewriter):
        elems = list(list_expr.elems)
        length = len(elems)

        total_bytes = 4 + 4 * length
        li_size = LIOp(total_bytes)
        malloc_ = CallOp("_malloc", [li_size], has_result=True)

        store_len = SWOp(LIOp(length), malloc_, 0)

        ops_ = [li_size, malloc_, store_len]
        offset = 4
        for e in elems:
            sw_ = SWOp(e, malloc_, offset)
            ops_.append(sw_)
            offset += 4

        # 最后必须产生 1 个结果(指向list的指针)
        rewriter.replace_op(list_expr, ops_, [malloc_])


class GetAddressPattern(RewritePattern):
    """
    get_address(value, index) => base + 4 + index*4
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_address: GetAddress, rewriter: PatternRewriter):
        val = get_address.value
        idx = get_address.index

        one = LIOp(1)
        plus_one = AddOp(idx, one)
        four = LIOp(4)
        times4 = MULOp(plus_one, four)
        final_addr = AddOp(val, times4)

        rewriter.replace_op(
            get_address,
            [one, plus_one, four, times4, final_addr],
            [final_addr],
        )


class IndexStringPattern(RewritePattern):
    """
    index_string =>
      1) check none
      2) check idx < 0 => _list_index_oob
      3) check idx >= length => _list_index_oob
      4) compute address: base + (idx+1)*4
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, indexString: IndexString, rewriter: PatternRewriter):
        val = indexString.value
        idx = indexString.index

        zero = LIOp(0)
        check_none = BEQOp(val, zero, "_list_index_none")

        # load length
        length = LWOp(val, 0)

        # 检查 idx < 0
        blt_neg = BLTOp(idx, zero, "_list_index_oob")
        # 检查 idx >= length
        check_oob = BGEOp(idx, length, "_list_index_oob")

        one = LIOp(1)
        plus_one = AddOp(idx, one)
        four = LIOp(4)
        times4 = MULOp(plus_one, four)
        final_ptr = AddOp(val, times4)

        rewriter.replace_op(
            indexString,
            [
                zero, check_none, length, blt_neg, check_oob,
                one, plus_one, four, times4, final_ptr
            ],
            [final_ptr],
        )


class AssignPattern(RewritePattern):
    """
    assign(target, value).
    If target is Memloc => store
    If target is register => move
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, assign_op: Assign, rewriter: PatternRewriter):
        target = assign_op.target
        val = assign_op.value
        if isinstance(target.type, MemlocType):
            sw_ = SWOp(val, target, 0)
            rewriter.replace_op(assign_op, [sw_])
        elif isinstance(target.type, RegisterType):
            zero = LIOp(0)
            mv_ = AddOp(val, zero)
            rewriter.replace_op(assign_op, [zero, mv_])
        else:
            raise NotImplementedError("Assign to unknown type target")


class MemberExprPattern(RewritePattern):
    """
    For object fields, not implemented in this pass
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, memexpr: MemberExpr, rewriter: PatternRewriter):
        raise NotImplementedError("MemberExpr not implemented in this pass.")


class ClassDefPattern(RewritePattern):
    """
    Not used or partial in this pass
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, clsdef: ClassDef, rewriter: PatternRewriter):
        raise NotImplementedError("ClassDef not implemented in this pass.")


class PassPattern(RewritePattern):
    """
    pass => no-op
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, pass_op: Pass, rewriter: PatternRewriter):
        rewriter.erase_matched_op(pass_op)


class YieldPattern(RewritePattern):
    """
    Remove yield
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, y_op: Yield, rewriter: PatternRewriter):
        rewriter.erase_matched_op(y_op)


class FuncDefPattern(RewritePattern):
    """
    func_def => riscv_ssa.func
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
    return => riscv_ssa.return
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, ret: Return, rewriter: PatternRewriter):
        val = ret.value
        new_ret = ReturnOp(val)
        rewriter.replace_op(ret, [new_ret])


# =========== 主 Pass ===========

class ChocoFlatToRISCVSSA(ModulePass):
    """
    Convert the choco_ir (flat) dialect to riscv_ssa dialect
    """
    name = "choco-flat-to-riscv-ssa"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # 1) Main pass
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
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

        # 2) Remove any yields
        walker2 = PatternRewriteWalker(
            GreedyRewritePatternApplier([YieldPattern()]),
            apply_recursively=True,
        )
        walker2.rewrite_module(op)
