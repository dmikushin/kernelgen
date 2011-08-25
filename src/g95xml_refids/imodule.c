
/* Internal module module
 * Copyright (C) 2000-2008 Free Software Foundation, Inc.
 * Contributed by Andy Vaught

This file is part of G95.

G95 is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

G95 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with G95; see the file COPYING.  If not, write to
the Free Software Foundation, 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.  */


#include "g95.h"
#include <string.h>



/* g95_internal_derived()-- Return nonzero if typespec is the internal
 * derived type. */

int g95_internal_derived(g95_typespec *ts, internal_type type) {

    return ts->type == BT_DERIVED && ts->derived->attr.itype == type;
}



/* derived parameter()-- Create a parameter of a derived type */

static void derived_parameter(char *name, internal_type itype,
			      internal_value ivalue) {
g95_expr *v;

    v = g95_get_expr();
    v->ts.type = BT_DERIVED;
    v->ts.derived = g95_current_ns->itypes[itype];

    v->type = EXPR_STRUCTURE;
    v->symbol = g95_current_ns->itypes[itype];
    v->where = g95_current_locus;
    v->value.constructor.c = NULL;
    v->value.constructor.ivalue = ivalue;

    g95_module_parameter(name, v);
}



/* char_parameter()-- Create a single-character parameter */

static void char_parameter(char *name, char value) {
g95_expr *v;

    v = g95_char_expr(1, g95_default_character_kind(), NULL);
    v->value.character.string[0] = value;

    g95_module_parameter(name, v);
}



/* integer_parameter()-- Create an integer parameter */

static void integer_parameter(char *name, int value) {

    g95_module_parameter(name, g95_int_expr(value));
}



/* use_iso_fortan_env()-- Use the ISO_FORTRAN_ENV module */

static void use_iso_fortran_env(void) {

    integer_parameter("character_storage_size", 8);
    integer_parameter("error_unit", 0);
    integer_parameter("file_storage_size", 8);
    integer_parameter("input_unit", 5);
    integer_parameter("iostat_end", -1);
    integer_parameter("iostat_eor", -2);
    integer_parameter("numeric_storage_size", 8*g95_default_integer_kind(0));
    integer_parameter("output_unit", 6);
    integer_parameter("stat_stopped_image", 217);

    g95_module_proc("this_image",   IPROC_THIS_IMAGE);
    g95_module_proc("num_images",   IPROC_NUM_IMAGES);
    g95_module_proc("co_lbound",    IPROC_CO_LBOUND);
    g95_module_proc("co_ubound",    IPROC_CO_UBOUND);
    g95_module_proc("image_index",  IPROC_IMAGE_INDEX);
}



/* use_iso_c_binding()-- Use the ISO_C_BINDING module */

static void use_iso_c_binding(void) {

    g95x_current_intrinsic_module_name = "iso_c_binding";

    integer_parameter("c_int",            sizeof(int));
    integer_parameter("c_short",          sizeof(short));
    integer_parameter("c_long",           sizeof(long));
    integer_parameter("c_long_long",      sizeof(long long));
    integer_parameter("c_signed_char",    sizeof(signed char));
    integer_parameter("c_size_t",         SIZEOF_SIZE_T);

    integer_parameter("c_int8_t",         1);
    integer_parameter("c_int16_t",        2);
    integer_parameter("c_int32_t",        4);
    integer_parameter("c_int64_t",        8);

    integer_parameter("c_int_least8_t",   1);
    integer_parameter("c_int_least16_t",  2);
    integer_parameter("c_int_least32_t",  4);
    integer_parameter("c_int_least64_t",  8);

    integer_parameter("c_int_fast8_t",    1);
    integer_parameter("c_int_fast16_t",   SIZEOF_SIZE_T);
    integer_parameter("c_int_fast32_t",   SIZEOF_SIZE_T);
    integer_parameter("c_int_fast64_t",   8);

    integer_parameter("c_intmax_t",       g95_default_integer_kind(0));
    integer_parameter("c_intptr_t",       SIZEOF_VOID_P);

    integer_parameter("c_float",         g95_default_real_kind(0));
    integer_parameter("c_double",        g95_default_double_kind());
    integer_parameter("c_long_double",   -1);

    integer_parameter("c_float_complex",          g95_default_real_kind(0));
    integer_parameter("c_double_complex",         g95_default_double_kind());
    integer_parameter("c_long_double_complex",    -1);

    integer_parameter("c_bool",           -1);

    integer_parameter("c_char",           g95_default_character_kind());

    char_parameter("c_null_char",        '\0');
    char_parameter("c_alert",            '\a');
    char_parameter("c_backspace",        '\b');
    char_parameter("c_form_feed",        '\f');
    char_parameter("c_new_line",         '\n');
    char_parameter("c_carriage_return",  '\r');
    char_parameter("c_horizontal_tab",   '\t');
    char_parameter("c_vertical_tab",     '\v');

    g95_module_type("c_ptr",    ITYPE_C_PTR);
    g95_module_type("c_funptr", ITYPE_C_FUNPTR);

    derived_parameter("c_null_ptr",    ITYPE_C_PTR,    IVALUE_C_NULL_PTR);
    derived_parameter("c_null_funptr", ITYPE_C_FUNPTR, IVALUE_C_NULL_FUNPTR);

    g95_module_proc("c_loc",            IPROC_C_LOC);
    g95_module_proc("c_funloc",         IPROC_C_FUNLOC);
    g95_module_proc("c_associated",     IPROC_C_ASSOCIATED);
    g95_module_proc("c_f_pointer",      IPROC_C_F_POINTER);
    g95_module_proc("c_f_procpointer",  IPROC_C_F_PROCPOINTER);
}



/* use_ieee_exceptions()-- Use the IEEE_EXCEPTIONS module */

static void use_ieee_exceptions(void) {

    g95x_current_intrinsic_module_name = "ieee_exceptions";

    g95_module_type("ieee_flag_type", ITYPE_IEEE_FLAG);

    derived_parameter("ieee_overflow", ITYPE_IEEE_FLAG, IVALUE_OVERFLOW);

    derived_parameter("ieee_divide_by_zero", ITYPE_IEEE_FLAG,
		      IVALUE_DIVIDE_BY_ZERO);

    derived_parameter("ieee_invalid", ITYPE_IEEE_FLAG, IVALUE_INVALID);

    derived_parameter("ieee_underflow", ITYPE_IEEE_FLAG, IVALUE_UNDERFLOW);

    derived_parameter("ieee_inexact", ITYPE_IEEE_FLAG, IVALUE_INEXACT);

    g95_module_type("ieee_status_type", ITYPE_IEEE_STATUS);

    g95_module_proc("ieee_support_flag", IPROC_IEEE_SUPPORT_FLAG);
    g95_module_proc("ieee_support_halting", IPROC_IEEE_SUPPORT_HALTING);
    g95_module_proc("ieee_get_flag", IPROC_IEEE_GET_FLAG);
    g95_module_proc("ieee_get_halting_mode", IPROC_IEEE_GET_HALTING_MODE);

    g95_module_proc("ieee_set_flag", IPROC_IEEE_SET_FLAG);
    g95_module_proc("ieee_set_halting_mode", IPROC_IEEE_SET_HALTING_MODE);
    g95_module_proc("ieee_get_status", IPROC_IEEE_GET_STATUS);
    g95_module_proc("ieee_set_status", IPROC_IEEE_SET_STATUS);
}



/* use_ieee_arithmetic()-- Use the IEEE_ARITHMETIC module */

static void use_ieee_arithemetic(void) {

    g95x_current_intrinsic_module_name = "ieee_arithemetic";

    g95_module_type("ieee_class_type", ITYPE_IEEE_CLASS);

    derived_parameter("ieee_signaling_nan",
		      ITYPE_IEEE_CLASS, IVALUE_SIGNALING_NAN);

    derived_parameter("ieee_quiet_nan", ITYPE_IEEE_CLASS, IVALUE_QUIET_NAN);

    derived_parameter("ieee_negative_inf",
		      ITYPE_IEEE_CLASS, IVALUE_NEGATIVE_INF);

    derived_parameter("ieee_negative_denormal",
		      ITYPE_IEEE_CLASS, IVALUE_NEGATIVE_DENORMAL);

    derived_parameter("ieee_negative_zero",
		      ITYPE_IEEE_CLASS, IVALUE_NEGATIVE_ZERO);

    derived_parameter("ieee_negative_normal",
		      ITYPE_IEEE_CLASS, IVALUE_NEGATIVE_NORMAL);

    derived_parameter("ieee_positive_inf",
		      ITYPE_IEEE_CLASS, IVALUE_POSITIVE_INF);

    derived_parameter("ieee_positive_denormal",
		      ITYPE_IEEE_CLASS, IVALUE_POSITIVE_DENORMAL);

    derived_parameter("ieee_positive_zero",
		      ITYPE_IEEE_CLASS, IVALUE_POSITIVE_ZERO);

    derived_parameter("ieee_positive_normal",
		      ITYPE_IEEE_CLASS, IVALUE_POSITIVE_NORMAL);

    derived_parameter("ieee_other_value",
		      ITYPE_IEEE_CLASS, IVALUE_OTHER_VALUE);

    g95_module_type("ieee_round_type", ITYPE_IEEE_ROUND);
    derived_parameter("ieee_nearest", ITYPE_IEEE_ROUND, IVALUE_ROUND_NEAREST);
    derived_parameter("ieee_to_zero", ITYPE_IEEE_ROUND, IVALUE_ROUND_TO_ZERO);
    derived_parameter("ieee_up", ITYPE_IEEE_ROUND, IVALUE_ROUND_UP);
    derived_parameter("ieee_down", ITYPE_IEEE_ROUND, IVALUE_ROUND_DOWN);

    g95_module_proc("ieee_support_datatype", IPROC_IEEE_SUPPORT_DATATYPE);
    g95_module_proc("ieee_support_denormal", IPROC_IEEE_SUPPORT_DENORMAL);
    g95_module_proc("ieee_support_divide", IPROC_IEEE_SUPPORT_DIVIDE);
    g95_module_proc("ieee_support_inf", IPROC_IEEE_SUPPORT_INF);
    g95_module_proc("ieee_support_nan", IPROC_IEEE_SUPPORT_NAN);
    g95_module_proc("ieee_support_rounding", IPROC_IEEE_SUPPORT_ROUNDING);
    g95_module_proc("ieee_support_sqrt", IPROC_IEEE_SUPPORT_SQRT);
    g95_module_proc("ieee_support_standard", IPROC_IEEE_SUPPORT_STANDARD);
    g95_module_proc("ieee_support_underflow_control",
		    IPROC_IEEE_SUPPORT_UNDERFLOW_CONTROL);

    g95_module_proc("ieee_class", IPROC_IEEE_CLASS);
    g95_module_proc("ieee_copy_sign", IPROC_IEEE_COPY_SIGN);
    g95_module_proc("ieee_is_finite", IPROC_IEEE_IS_FINITE);
    g95_module_proc("ieee_is_nan", IPROC_IEEE_IS_NAN);
    g95_module_proc("ieee_is_negative", IPROC_IEEE_IS_NEGATIVE);
    g95_module_proc("ieee_is_normal", IPROC_IEEE_IS_NORMAL);
    g95_module_proc("ieee_logb", IPROC_IEEE_LOGB);
    g95_module_proc("ieee_next_after", IPROC_IEEE_NEXT_AFTER);
    g95_module_proc("ieee_rem", IPROC_IEEE_REM);
    g95_module_proc("ieee_rint", IPROC_IEEE_RINT);
    g95_module_proc("ieee_scalb", IPROC_IEEE_SCALB);
    g95_module_proc("ieee_unordered", IPROC_IEEE_UNORDERED);
    g95_module_proc("ieee_value", IPROC_IEEE_VALUE);
    g95_module_proc("ieee_get_rounding_mode", IPROC_IEEE_GET_ROUNDING_MODE);
    g95_module_proc("ieee_get_underflow_mode", IPROC_IEEE_GET_UNDERFLOW_MODE);
    g95_module_proc("ieee_set_rounding_mode", IPROC_IEEE_SET_ROUNDING_MODE);
    g95_module_proc("ieee_set_underflow_mode", IPROC_IEEE_SET_UNDERFLOW_MODE);
    g95_module_proc("ieee_selected_real_kind", IPROC_IEEE_SELECTED_REAL_KIND);

    use_ieee_exceptions();
}



/* use_ieee_features()-- Use the IEEE_FEATURES module */

static void use_ieee_features(void) {

    g95x_current_intrinsic_module_name = "ieee_features";

    g95_module_type("ieee_features_type", ITYPE_IEEE_FEATURES);

    derived_parameter("ieee_datatype", ITYPE_IEEE_FEATURES, IVALUES_DATATYPE);

    derived_parameter("ieee_denormal", ITYPE_IEEE_FEATURES, IVALUES_DENORMAL);

    derived_parameter("ieee_divide", ITYPE_IEEE_FEATURES, IVALUES_DIVIDE);

    derived_parameter("ieee_halting", ITYPE_IEEE_FEATURES, IVALUES_HALTING);

    derived_parameter("ieee_inexact_flag",
		      ITYPE_IEEE_FEATURES, IVALUES_INEXACT_FLAG);

    derived_parameter("ieee_inf", ITYPE_IEEE_FEATURES, IVALUES_INF);

    derived_parameter("ieee_invalid_flag",
		      ITYPE_IEEE_FEATURES, IVALUES_INVALID_FLAG);

    derived_parameter("ieee_nan", ITYPE_IEEE_FEATURES, IVALUES_NAN);

    derived_parameter("ieee_rounding", ITYPE_IEEE_FEATURES, IVALUES_ROUNDING);

    derived_parameter("ieee_sqrt", ITYPE_IEEE_FEATURES, IVALUES_SQRT);

    derived_parameter("ieee_underflow_flag",
		      ITYPE_IEEE_FEATURES, IVALUES_UNDERFLOAT_FLAG);
}



/* use_g95()-- Use the g95 internal library module */

static void use_g95(void) {


}



/* g95_use_internal()-- Use an internal module.  Returns nonzero if
 * the module isn't found. */

int g95_use_internal(char *name) {

    if (strcmp(name, "iso_fortran_env") == 0)
	use_iso_fortran_env();

    else if (strcmp(name, "iso_c_binding") == 0)
	use_iso_c_binding();

    else if (strcmp(name, "ieee_exceptions") == 0)
	use_ieee_exceptions();

    else if (strcmp(name, "ieee_arithmetic") == 0)
	use_ieee_arithemetic();

    else if (strcmp(name, "ieee_features") == 0)
	use_ieee_features();

    else if (strcmp(name, "g95") == 0)
	use_g95();

    else
	return 1;

    return 0;
}

