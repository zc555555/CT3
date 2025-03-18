// RUN: choco-opt "%s" -t riscv | filecheck "%s"

builtin.module {
  "riscv.addi"() <{"rd" = !riscv.reg<a0>, "rs1" = !riscv.reg<zero>, "immediate" = 0 : i64}> : () -> ()
  // CHECK: addi a0, zero, 0

  "riscv.addi"() <{"rd" = !riscv.reg<a7>, "rs1" = !riscv.reg<zero>, "immediate" = 93 : i64}> : () -> ()
  // CHECKT-NEXT: addi a7, zero, 93

  "riscv.ecall"() : () -> ()
  // CHECKT-NEXT: ecall
}
