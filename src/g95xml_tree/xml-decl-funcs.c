g95_expr *xbind_expr = NULL;

static void
set_bind_expr (g95_expr * e)
{
  if (xbind_expr)
    g95x_die ("");
  xbind_expr = e;
}

static void
clear_bind_expr ()
{
  if (xbind_expr)
    g95_free_expr (xbind_expr);
  xbind_expr = NULL;
}

g95_expr *
g95x_take_bind ()
{
  g95_expr *e = xbind_expr;
  xbind_expr = NULL;
  return e;
}

static g95_expr *
get_bind_expr ()
{
  return xbind_expr;
}

/*
  1 match_bind_name

  2 match_rest_bind_attr
  3 match_attr_spec
  4 g95_match_bind

  2 match_bind
  3 match_suffix
  4 g95_match_function
  4 g95_match_entry
  4 g95_match_subroutine
*/

g95_typespec *
g95x_get_current_ts ()
{
  return &current_ts;
}

symbol_attribute *
g95x_get_current_attr ()
{
  return &current_attr;
}

/*  */

static g95x_voidp_list *decl_list = NULL;

static void
g95x_decl_push_list_symbol (g95_symbol * s)
{
  return g95x_push_list_symbol (&decl_list, s);
}

static void
g95x_decl_set_dimension (void *s)
{
  return g95x_set_dimension (&decl_list, s);
}

static void
g95x_decl_set_length (void *s)
{
  return g95x_set_length (&decl_list, s);
}

static void
g95x_decl_set_initialization (g95_symbol * s)
{
  return g95x_set_initialization (&decl_list, s);
}

static void
g95x_decl_push_list_generic (g95_symtree * s)
{
  return g95x_push_list_generic (&decl_list, s);
}

static void
g95x_decl_push_list_user_op (g95_symtree * s)
{
  return g95x_push_list_user_op (&decl_list, s);
}

static void
g95x_decl_push_list_intr_op_idx (int s)
{
  return g95x_push_list_intr_op_idx (&decl_list, s);
}

static void
g95x_decl_push_list_component (g95_component * s)
{
  return g95x_push_list_component (&decl_list, s);
}

static void
g95x_decl_push_list_common (g95_common_head * c)
{
  return g95x_push_list_common (&decl_list, c);
}

static void
g95x_decl_set_locus (g95x_locus * xw)
{
  if (!g95x_option.enable)
    return;
  if (decl_list)
    {
      decl_list->where = *xw;
    }
  else
    {
      g95x_die ("");
    }
}

static void
g95x_decl_free_list ()
{
  return g95x_free_list (&decl_list);
}

g95x_voidp_list *
g95x_decl_take_list ()
{
  return g95x_take_list (&decl_list);
}

g95_array_spec *
g95x_take_current_as ()
{
  g95_array_spec *as = current_as;
  current_as = NULL;
  return as;
}


g95_coarray_spec *
g95x_take_current_cas ()
{
  g95_coarray_spec *cas = current_cas;
  current_cas = NULL;
  return cas;
}


static g95x_ext_locus *exw = NULL;


static void
set_dimension (g95_array_spec * asA, g95_array_spec * asB, void *s)
{
  if (asA && asA->x.alias)
    asA = asA->x.alias;
  if (asB && asB->x.alias)
    asB = asB->x.alias;
  if (asA != asB)
    g95x_decl_set_dimension (s);
}

static void
set_length (g95_charlen * clA, g95_charlen * clB, void *s)
{
  if (clA && clA->x.alias)
    clA = clA->x.alias;
  if (clB && clB->x.alias)
    clB = clB->x.alias;
  if (clA != clB)
    g95x_decl_set_length (s);
}
