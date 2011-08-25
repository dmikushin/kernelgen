g95_expr *xbind_expr = NULL;

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

g95_typespec* g95x_get_current_ts() {
  return &current_ts;
}

symbol_attribute* g95x_get_current_attr() {
  return &current_attr;
}

/*  */

static g95x_voidp_list* decl_list = NULL;

static void g95x_push_decl_list_symbol( g95_symbol *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.symbol = s;
  sl->type = G95X_VOIDP_SYMBOL;
  sl->next = decl_list;
  decl_list = sl;
}

static void g95x_set_dimension( g95_symbol *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = decl_list;
  for( ; sl; sl = sl->next ) {
    if( sl->type == G95X_VOIDP_SYMBOL ) 
      if( sl->u.symbol == s ) {
        sl->dimension = 1;
        return;
      }
  }
  g95x_die( "Cannot find symbol %R in decl_list" );
}

static void g95x_set_initialization( g95_symbol *s ) {
  g95x_voidp_list* sl = decl_list;
  if( ! g95x_option.enable )
    return;
  for( ; sl; sl = sl->next ) {
    if( sl->type == G95X_VOIDP_SYMBOL ) 
      if( sl->u.symbol == s ) {
        sl->init = 1;
        return;
      }
  }
  g95x_die( "Cannot find symbol %R in decl_list" );
}

static void g95x_push_decl_list_generic( g95_symtree *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.generic = s;
  sl->type = G95X_VOIDP_GENERIC;
  sl->next = decl_list;
  decl_list = sl;
}

static void g95x_push_decl_list_user_op( g95_symtree *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.userop = s;
  sl->type = G95X_VOIDP_USEROP;
  sl->next = decl_list;
  decl_list = sl;
}

/*
static void g95x_push_decl_list_intr_op( g95_interface *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.introp = s;
  sl->type = G95X_VOIDP_INTROP;
  sl->next = decl_list;
  decl_list = sl;
}
*/

static void g95x_push_decl_list_intr_op_idx( int s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.introp_idx = s;
  sl->type = G95X_VOIDP_INTROP_IDX;
  sl->next = decl_list;
  decl_list = sl;
}

static void g95x_push_decl_list_component( g95_component *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.component = s;
  sl->type = G95X_VOIDP_COMPNT;
  sl->next = decl_list;
  decl_list = sl;
}

static void g95x_push_decl_list_common( g95_common_head *c ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.common = (struct g95_common_head *)c;
  sl->type = G95X_VOIDP_COMMON;
  sl->next = decl_list;
  decl_list = sl;
}

/*
static void g95x_push_decl_list_k( int k ) {
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->voidp = NULL;
  sl->next = decl_list;
  decl_list = sl;
}
*/

static void g95x_free_decl_list() {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list *sl, *sl1;
  for( sl = decl_list; sl; ) {
    sl1 = sl->next;
    g95_free( sl );
    sl = sl1;
  }
  decl_list = NULL;
}

g95x_voidp_list* g95x_take_decl_list() {
  g95x_voidp_list* sl = decl_list;
  decl_list = NULL;
  return sl;
}

g95_array_spec* g95x_take_current_as() {
  g95_array_spec* as = current_as;
  current_as = NULL;
  return as;
}


g95_coarray_spec* g95x_take_current_cas() {
  g95_coarray_spec* cas = current_cas;
  current_cas = NULL;
  return cas;
}


static g95x_ext_locus *exw = NULL;

