#!/usr/bin/env python3

import argparse
from typing import Literal, Optional, TextIO

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp

import riscv.dialect as riscv
from riscv.lexer import Lexer, Token, TokenClass
from riscv.printer import *

# MLContext, containing information about the registered dialects
context = MLContext()

context.load_dialect(riscv.RISCV)


class Operand:
    def __init__(self, name: Optional[str], immediate: int | None = None):
        self.name = name
        self.immediate = immediate

    def __repr__(self):
        s = "Operand("

        if self.name:
            s += '"' + self.name + '"'
        else:
            s += "None"

        if self.immediate:
            s += ", " + str(self.immediate)

        s += ")"
        return s


class Parser:
    def __init__(self, stream: TextIO) -> None:
        self.lexer = Lexer(stream)

    def token(self) -> Token | None:
        return self.lexer.peek()

    def isType(self, ty: TokenClass) -> bool:
        t = self.token()
        return t is not None and t.type == ty

    def match(self, ty: TokenClass) -> Token:
        t = self.lexer.next()
        if t is None or t.type != ty:
            raise Exception("Error: unexpected token type!")

        return t

    def tryMatch(self, ty: TokenClass) -> Token | Literal[False]:
        if self.isType(ty):
            t = self.match(ty)
            return t

        return False

    def parseOperand(self):
        if self.tryMatch(TokenClass.DOT):
            symbol = self.match(TokenClass.SYMBOL)
            return Operand("." + symbol.value)

        if symbol := self.tryMatch(TokenClass.SYMBOL):
            return Operand(symbol.value)

        if literal := self.tryMatch(TokenClass.LITERAL):
            return Operand(literal.value)

        immediate = ""
        if self.tryMatch(TokenClass.MINUS):
            immediate = "-"

        immediate += self.match(TokenClass.INTEGER).value

        symbol = None

        if self.tryMatch(TokenClass.LPAREN):
            symbol = self.match(TokenClass.SYMBOL).value
            self.match(TokenClass.RPAREN)

        return Operand(symbol, int(immediate))

    def parse_(self):
        while self.lexer.peek():
            if self.isType(TokenClass.DOT):
                self.match(TokenClass.DOT)
                name = self.match(TokenClass.SYMBOL).value

                if self.tryMatch(TokenClass.NEWLINE):
                    yield riscv.DirectiveOp(name)
                    continue

                op = self.parseOperand()

                assert op.name

                yield riscv.DirectiveOp(name, op.name)

            elif self.isType(TokenClass.SYMBOL):
                name = self.match(TokenClass.SYMBOL).value

                if self.tryMatch(TokenClass.COLON):
                    yield riscv.LabelOp(name)
                    continue

                op_def = context.get_op("riscv." + name)

                if self.tryMatch(TokenClass.NEWLINE):
                    if issubclass(op_def, RiscvNoParamsOperation):
                        yield op_def()
                        continue
                    else:
                        raise Exception(f"Unknown operation {name}")

                ops = [self.parseOperand()]

                while self.tryMatch(TokenClass.COMMA):
                    op = self.parseOperand()
                    ops.append(op)

                assert ops[0].immediate == None

                if issubclass(op_def, Riscv1Rd1ImmOperation):
                    assert len(ops) == 2
                    assert ops[0].immediate == None
                    assert ops[1].name == None

                    yield op_def(ops[0].name, ops[1].immediate)
                elif issubclass(op_def, Riscv1Rs1OffOperation) or issubclass(
                    op_def, Riscv1Rd1OffOperation
                ):
                    assert len(ops) == 2
                    assert ops[0].immediate == None
                    assert ops[1].name == None or ops[1].immediate == None

                    if ops[1].name:
                        yield op_def(ops[0].name, ops[1].name)
                    else:
                        yield op_def(ops[0].name, ops[1].immediate)
                elif issubclass(op_def, Riscv1Rd2RsOperation):
                    assert ops[1].immediate == None
                    assert ops[2].immediate == None

                    yield op_def(ops[0].name, ops[1].name, ops[2].name)
                elif issubclass(op_def, Riscv1Rd1Rs1ImmOperation) or issubclass(
                    op_def, Riscv2Rs1ImmOperation
                ):
                    assert len(ops) in [2, 3]
                    if len(ops) == 3:
                        assert ops[2].name == None
                        assert ops[1].immediate == None
                        immediate = ops[2].immediate
                    else:
                        immediate = ops[1].immediate

                    if immediate is None:
                        immediate = "0"

                    immediate_int = int(immediate)  # type: ignore

                    yield op_def(ops[0].name, ops[1].name, immediate_int)
                elif (
                    issubclass(op_def, Riscv2Rs1OffOperation)
                    or issubclass(op_def, Riscv1Rd1Rs1OffOperation)
                    or issubclass(op_def, Riscv1Rs1Rt1OffOperation)
                ):
                    assert len(ops) in [2, 3]
                    if len(ops) == 3:
                        assert ops[1].immediate == None
                        assert ops[2].name == None or ops[2].immediate == None

                        if ops[2].name:
                            yield op_def(ops[0].name, ops[1].name, ops[2].name)
                            continue

                        immediate = ops[2].immediate
                    else:
                        immediate = ops[1].immediate

                    immediate_int = int(immediate)  # type: ignore

                    yield op_def(ops[0].name, ops[1].name, immediate_int)
                elif issubclass(op_def, Riscv1Rd1RsOperation):
                    assert len(ops) == 2
                    assert ops[0].immediate == None

                    yield op_def(ops[0].name, ops[1].name)
                else:
                    raise Exception(f"Unknown operation {name}")
            else:
                self.match(TokenClass.NEWLINE)

    def parse(self) -> ModuleOp:
        return ModuleOp([instruction for instruction in self.parse_()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A RISC-V interpreter")
    parser.add_argument("file", type=argparse.FileType("r"))
    parser.add_argument(
        "--format",
        choices=[
            "asm",
        ],
        default="asm",
    )
    args = parser.parse_args()

    module = Parser(args.file).parse()
    print_program(list(module.ops), args.format)
