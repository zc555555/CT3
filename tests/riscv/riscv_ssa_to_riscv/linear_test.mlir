// RUN: choco-opt "%s" -p riscv-ssa-to-riscv,riscv-function-lowering -t riscv -o "%t" && riscv-interpreter "%t" | filecheck "%s"

builtin.module {
  "riscv_ssa.func"() <{"func_name" = "_main"}> ({
    %0 = "riscv_ssa.li"() <{"immediate" = 93 : i64}> : () -> !riscv_ssa.reg
    %1 = "riscv_ssa.li"() <{"immediate" = 42 : i64}> : () -> !riscv_ssa.reg
    "riscv_ssa.ecall"(%0, %1): (!riscv_ssa.reg, !riscv_ssa.reg) -> ()
  }) : () -> ()
}

// CHECK: Return code: 42
