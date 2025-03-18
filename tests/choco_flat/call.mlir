// RUN: choco-opt "%s" | filecheck "%s"

//
// print(input())
//

builtin.module {
  %0 = "choco_ir.call_expr"() <{"func_name" = "input"}> : () -> !choco_ir.named_type<"int">
  "choco_ir.call_expr"(%0) <{"func_name" = "print"}> : (!choco_ir.named_type<"int">) -> ()
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "choco_ir.call_expr"() <{"func_name" = "input"}> : () -> !choco_ir.named_type<"int">
// CHECK-NEXT:   "choco_ir.call_expr"(%0) <{"func_name" = "print"}> : (!choco_ir.named_type<"int">) -> ()
// CHECK-NEXT: }
