/* common */

g95x_voidp_list *common_list = NULL;

static void g95x_push_common_list_symbol( g95_symbol *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.symbol = s;
  sl->type = G95X_VOIDP_SYMBOL;
  sl->next = common_list;
  common_list = sl;
}

static void g95x_set_dimension( g95_symbol *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = common_list;
  for( ; sl; sl = sl->next ) {
    if( sl->type == G95X_VOIDP_SYMBOL ) 
      if( sl->u.symbol == s ) {
        sl->dimension = 1;
        return;
      }
  }
  g95x_die( "Cannot find symbol %R in common_list" );
}

static void g95x_push_common_list_common( g95_common_head *c ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.common = (struct g95_common_head*)c;
  sl->type = G95X_VOIDP_COMMON;
  sl->next = common_list;
  common_list = sl;
}

static void g95x_free_common_list() {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list *sl, *sl1;
  for( sl = common_list; sl; ) {
    sl1 = sl->next;
    g95_free( sl );
    sl = sl1;
  }
  common_list = NULL;
}

g95x_voidp_list* g95x_take_common_list() {
  g95x_voidp_list* sl = common_list;
  common_list = NULL;
  return sl;
}


/* enumerator */


g95x_voidp_list *enumerator_list = NULL;

static void g95x_push_enumerator_list_symbol( g95_symbol *s ) {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list* sl = g95x_get_voidp_list();
  sl->u.symbol = s;
  sl->type = G95X_VOIDP_SYMBOL;
  sl->next = enumerator_list;
  enumerator_list = sl;
}

static void g95x_free_enumerator_list() {
  if( ! g95x_option.enable )
    return;
  g95x_voidp_list *sl, *sl1;
  for( sl = enumerator_list; sl; ) {
    sl1 = sl->next;
    g95_free( sl );
    sl = sl1;
  }
  enumerator_list = NULL;
}

g95x_voidp_list* g95x_take_enumerator_list() {
  g95x_voidp_list* sl = enumerator_list;
  enumerator_list = NULL;
  return sl;
}


int g95x_enum_bindc() {
  return enum_bindc;
}

