# type: ignore

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from choco.dialects.choco_flat import *
from riscv.ssa_dialect import *


class LiteralPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Literal, rewriter: PatternRewriter):
        value = op.value
        if isinstance(value, IntegerAttr):
            constant = LIOp(op.value)
            rewriter.replace_op(op, [constant])
            return

        if isinstance(value, BoolAttr):
            raise NotImplementedError()

        if isinstance(value, NoneAttr):
            raise NotImplementedError()

        if isinstance(value, StringAttr):
            raise NotImplementedError()

        return


class CallPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallExpr, rewriter: PatternRewriter):
        if op.func_name.data == "len":
            zero = LIOp(0)
            maybe_fail = BEQOp(op.args[0], zero, f"_error_len_none")
            read_size = LWOp(op.args[0], 0)
            rewriter.replace_op(op, [zero, maybe_fail, read_size])
            return

        call = CallOp(op.func_name, op.args, has_result=bool(len(op.results)))
        rewriter.replace_op(op, [call])


class AllocPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: Alloc, rewriter: PatternRewriter):
        raise NotImplementedError()


class StorePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, store_op: Store, rewriter: PatternRewriter):
        raise NotImplementedError()


class LoadPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, load_op: Load, rewriter: PatternRewriter):
        raise NotImplementedError()


class UnaryExprPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, unary_op: UnaryExpr, rewriter: PatternRewriter):
        raise NotImplementedError()


class BinaryExprPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, bin_op: BinaryExpr, rewriter: PatternRewriter):
        raise NotImplementedError()


class IfPattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, if_op: If, rewriter: PatternRewriter):
        raise NotImplementedError()


class AndPattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, and_op: EffectfulBinaryExpr, rewriter: PatternRewriter):
        raise NotImplementedError()


class OrPattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, or_op: EffectfulBinaryExpr, rewriter: PatternRewriter):
        raise NotImplementedError()


class IfExprPattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, if_op: IfExpr, rewriter: PatternRewriter):
        raise NotImplementedError()


class WhilePattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, while_op: While, rewriter: PatternRewriter):
        raise NotImplementedError()


class ListExprPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, list_expr: ListExpr, rewriter: PatternRewriter):
        raise NotImplementedError()


class GetAddressPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_address: GetAddress, rewriter: PatternRewriter):
        raise NotImplementedError()


class IndexStringPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, indexString: IndexString, rewriter: PatternRewriter):
        raise NotImplementedError()


class YieldPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_address: Yield, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class FuncDefPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func: FuncDef, rewriter: PatternRewriter):
        new_func = FuncOp.create(
            result_types=[],
            properties={"func_name": StringAttr(func.func_name.data)},
        )

        new_region = rewriter.move_region_contents_to_new_regions(func.func_body)
        new_func.add_region(new_region)
        for arg in new_region.block.args:
            rewriter.modify_value_type(arg, RegisterType())

        rewriter.replace_op(func, new_func)


class ReturnPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, ret: Return, rewriter: PatternRewriter):
        raise NotImplementedError()


class ChocoFlatToRISCVSSA(ModulePass):
    name = "choco-flat-to-riscv-ssa"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LiteralPattern(),
                    CallPattern(),
                    UnaryExprPattern(),
                    BinaryExprPattern(),
                    StorePattern(),
                    LoadPattern(),
                    AllocPattern(),
                    IfPattern(),
                    AndPattern(),
                    OrPattern(),
                    IfExprPattern(),
                    WhilePattern(),
                    ListExprPattern(),
                    GetAddressPattern(),
                    IndexStringPattern(),
                    FuncDefPattern(),
                    ReturnPattern(),
                ]
            ),
            apply_recursively=True,
        )

        walker.rewrite_module(op)

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    YieldPattern(),
                ]
            ),
            apply_recursively=True,
        )

        walker.rewrite_module(op)
