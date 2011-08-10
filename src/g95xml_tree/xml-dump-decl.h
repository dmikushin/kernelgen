#ifndef _XML_DUMP_DECL_H
#define _XML_DUMP_DECL_H

#include <stdlib.h>

typedef struct g95x_stop_code {
  g95x_locus where; 
  int code;
} g95x_stop_code;

typedef struct g95x_intrinsic_type_name {
  g95x_locus where;
  const char* name;
} g95x_intrinsic_type_name;
 
typedef struct g95x_global_symbol {
  const char* name;
  int defined;
  g95_symbol* sym;
} g95x_global_symbol;

typedef struct g95x_global_common {
  const char* name;
  g95_common_head* com;
} g95x_global_common;

typedef struct g95x_section_subscript
{
  g95_ref *ref;
  int index;
} g95x_section_subscript;

typedef struct g95x_generic
{
  enum
  {
    G95X_GENERIC_GNRC,
    G95X_GENERIC_USOP,
    G95X_GENERIC_INOP
  } type;
  g95_intrinsic_op inop;
  union
  {
    g95_interface **inop;
    g95_symtree *st_gen;
    g95_symtree *st_uop;
  } u;
  g95_interface *intf;
  g95_interface *intf_list;
  g95_access access;
} g95x_generic;

typedef struct g95x_locus_section
{
  g95x_locus *xr;
} g95x_locus_section;

typedef struct g95x_comment
{
  g95_linebuf *lb;
  int column;
} g95x_comment;

typedef struct g95x_label_ref
{
  const void *addr;
  g95_st_label *label;
  g95x_locus *xw;
} g95x_label_ref;

typedef struct g95x_shape_spec
{
  g95_array_spec *as;
  int i;
} g95x_shape_spec;

typedef struct g95x_alloc_spec
{
  g95_expr *lb, *ub;
  g95x_locus where;
} g95x_alloc_spec;

typedef struct g95x_co_alloc_spec
{
  g95_expr *lb, *ub;
  g95x_locus where;
} g95x_co_alloc_spec;

typedef struct g95x_coshape_spec
{
  g95_coarray_spec *cas;
  int i;
  g95x_locus where;
} g95x_coshape_spec;

typedef struct g95x_symbol_ref
{
  const void *sym;
  g95x_locus *xw;
  char als[2 * G95_MAX_SYMBOL_LEN + 1];
  const char *cls;
  int strcmp;
  const char *type;
} g95x_symbol_ref;

typedef struct g95x_user_operator1_ref
{
  g95x_symbol_ref sr;
  g95x_locus xwo;
} g95x_user_operator1_ref;

typedef struct g95x_intrinsic_operator1_ref
{
  g95x_symbol_ref sr;
  g95x_locus xwo;
} g95x_intrinsic_operator1_ref;

typedef struct g95x_assignment1_ref
{
  g95x_symbol_ref sr;
  g95x_locus xwo;
} g95x_assignment1_ref;


typedef struct g95x_operator
{
  g95_intrinsic_op iop;
  g95_namespace *ns;
} g95x_operator;

typedef struct g95x_expr_chst
{
  const void *addr;
  g95x_locus *where;
} g95x_expr_chst;

typedef struct g95x_alt_return
{
  const void *addr;
  g95x_locus *where;
} g95x_alt_return;

typedef struct g95x_io_control_spec
{
  enum
  {
    G95X_CS_EXPR,
    G95X_CS_EXCH,
    G95X_CS_LABL,
    G95X_CS_SMBL
  } type;
  union
  {
    g95_expr **expr;
    g95_st_label **label;
    g95_symbol **symbol;
    void *p;
  } u;
  g95x_locus *xw;
  g95x_locus where;
  const char *name;
  const char *class;
} g95x_io_control_spec;

typedef struct g95x_attr_spec
{
  g95x_locus where;
  const char *name;
  g95_array_spec *as;
  g95_expr *bind;
  g95_symbol *result;
  g95x_locus *result_where;
} g95x_attr_spec;

typedef struct g95x_executable_program
{
  g95_namespace *head;
  g95x_locus where;
} g95x_executable_program;

#define g95x_get_attr_spec() g95_getmem( sizeof( g95x_attr_spec ) )

typedef void (*g95x_dumper) (const void *);

typedef struct g95x_item_locus
{
  struct g95x_item_locus *next;
  const char *key;
  g95x_locus *xwl;
  g95x_dumper dmp;
  const void *obj;
} g95x_item_locus;

#define g95x_get_item_locus() g95_getmem( sizeof( g95x_item_locus ) )

typedef struct g95x_temp
{
  struct g95x_temp *next;
  const void *obj;
} g95x_temp;

#define g95x_get_temp() g95_getmem( sizeof( g95x_temp ) )

typedef struct g95x_letter_spec
{
  char c1, c2;
  g95x_locus where;
  g95x_locus where1;
} g95x_letter_spec;

enum
{
  G95X_DUMP_ATTR_ALL = 0x0003,
  G95X_DUMP_ATTR_DEC = 0x0001,
  G95X_DUMP_ATTR_SUF = 0x0002
};

g95_namespace *g95x_dump_ns;

#define simple_statement( sta ) \
  ( ( (sta)->type == ST_SIMPLE_IF ) \
 || ( (sta)->type == ST_WHERE     ) \
 || ( (sta)->type == ST_FORALL   ) )

#include "xml-cpp.h"

g95x_locus* g95x_get_data_statement_set_locus (g95_data*);
g95x_locus* g95x_get_data_var_locus (g95_data_variable*);
g95x_locus* g95x_get_data_val_locus (g95_data_value*);
g95x_locus* g95x_get_iterator_locus (g95_iterator*);
g95x_locus* g95x_get_common_block_decl_locus (g95x_voidp_list*);
g95x_locus* g95x_get_equivalence_set_locus (g95_equiv*);
g95x_locus* g95x_get_namelist_group_locus (g95x_voidp_list*);
g95x_locus* g95x_get_forall_iterator_locus (g95_forall_iterator*);
g95x_locus* g95x_get_constructor_locus (g95_constructor*);

#define G95X_FULLIFY( e ) \
  do { if( (e) && (e)->x.full ) e = (e)->x.full; } \
  while( 0 )


#endif
