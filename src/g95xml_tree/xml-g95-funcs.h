#include "xml-dump-decl.h"



/* parse.c */

g95_symbol *g95x_stmt_func_take_symbol ();
g95x_locus *g95x_get_current_statement_locus ();

/* decl.c */

g95_typespec *g95x_get_current_ts ();
symbol_attribute *g95x_get_current_attr ();
g95x_voidp_list *g95x_decl_take_list ();
g95_array_spec *g95x_take_current_as ();
g95_expr *g95x_take_bind ();
g95_array_spec *g95x_take_current_as ();
g95_coarray_spec *g95x_take_current_cas ();

/* symbol.c */

g95_typespec *g95x_get_implicit_ts ();
g95x_implicit_spec *g95x_take_implicit_spec_list ();

/* intrinsic.c */

void g95x_get_intrinsics_info (g95_intrinsic_sym **, g95_intrinsic_sym **,
			       int *, int *);
void g95x_mark_intrinsic_used (g95_intrinsic_sym * is);

/* scanner.c */

int g95x_current_line (g95_file *);

g95_linebuf *g95x_get_line_head ();
g95_linebuf *g95x_get_line_tail ();
g95_file *g95x_get_file_head ();
g95_file *g95x_get_file_top ();
int g95x_get_line_width ();

/* arith.c */

extern int g95x_simplify_state[4096];
extern int g95x_simplify_index;

#define g95x_simplify_none() g95x_simplify_state[g95x_simplify_index]
#define g95x_simplify_push( x ) \
  do { g95x_simplify_state[++g95x_simplify_index] = x; } while( 0 )
#define g95x_simplify_pop() --g95x_simplify_index
#define g95x_simplify_init() g95x_simplify_push( 1 )

/* match.c */

int g95x_enum_bindc ();
g95x_voidp_list *g95x_common_take_list ();
g95x_voidp_list *g95x_enumerator_take_list ();
g95x_voidp_list *g95x_naml_take_list ();
g95x_voidp_list *g95x_import_take_list ();


/* expr.c */

g95x_kind *g95x_get_kind (g95_namespace *);

/* module.c */

int g95x_get_only_flag ();
g95_use_rename *g95x_module_take_rename_list ();
g95x_voidp_list g95x_module_get_mod ();

/* xml-basic.c */

int g95x_init_options (int argc, char *argv[]);
int g95x_parse_arg (int, char *argv[]);

int g95x_valid_expr (g95_expr *);
try g95x_simplify_expr (g95_expr ** expr);
try g95x_simplify_spec_expr (g95_expr ** e);

void g95x_push_statement (g95_statement, g95_code *, g95x_locus *);

void g95x_push_locus (g95x_locus *, g95x_locus_type);
void g95x_push_locus_mmbr (g95x_locus *, g95_symbol * derived_type);
void g95x_push_locus_labl (g95x_locus * loc, g95_st_label * label);
void g95x_push_locus_usop (g95x_locus * loc, g95_expr * e);
void g95x_push_locus_inop (g95x_locus * loc, g95_intrinsic_op op);

g95x_ext_locus *g95x_get_current_ext_locus_ref ();

g95x_locus g95x_pop_locus ();

void g95x_get_last_locus (g95x_locus *, g95x_locus_type, int);

void g95x_mark_1 (g95x_locus *);
void g95x_mark_2 (g95x_locus *);
void g95x_mark_1_l (g95x_locus * xw, g95_locus * w);
void g95x_mark_2_l (g95x_locus * xw, g95_locus * w);

int g95x_locus_valid (g95x_locus * xw);



int g95x_init ();
int g95x_done ();

void g95x_error (int, int, const char *, const char *, ...);



void g95x_free_namespace (g95_namespace * ns);

void g95x_fixup_generics (g95_namespace *);


extern g95x_statement *g95x_statement_head, *g95x_statement_tail;

extern const char *g95x_current_intrinsic_module_name;
extern char **g95x_builtin_macros;

void g95x_push_list_symbol (g95x_voidp_list ** list, g95_symbol * s);
void g95x_push_list_common (g95x_voidp_list ** list, g95_common_head * c);
void g95x_set_dimension (g95x_voidp_list ** list, void *s);
void g95x_free_list (g95x_voidp_list ** list);
g95x_voidp_list *g95x_take_list (g95x_voidp_list ** list);
void g95x_set_length (g95x_voidp_list ** list, void *s);
void g95x_set_initialization (g95x_voidp_list ** list, g95_symbol * s);
void g95x_push_list_generic (g95x_voidp_list ** list, g95_symtree * s);
void g95x_push_list_user_op (g95x_voidp_list ** list, g95_symtree * s);
void g95x_push_list_intr_op_idx (g95x_voidp_list ** list, int s);
void g95x_push_list_component (g95x_voidp_list ** list, g95_component * s);


const void *g95x_resolve_intrinsic_operator (g95_intrinsic_op);
const void *g95x_resolve_generic (const char *name);
const void *g95x_resolve_user_operator (const char *name);


/* xml-print.c */

#include <string.h>

const char *REFFMT0;
const char *REFFMT1;


int g95x_start ();
int g95x_end ();



#define g95x_warn( ... ) \
    g95x_error( 0, __LINE__, __FILE__, __VA_ARGS__ )

#define g95x_die( ... ) \
    g95x_error( 1, __LINE__, __FILE__, __VA_ARGS__ )

void g95x_resolve_gen (char *name, g95_namespace * ns, void **p);
void g95x_resolve_uop (char *name, g95_namespace * ns, void **p);
void g95x_resolve_iop (g95_intrinsic_op type, g95_namespace * ns, void **p);
void g95x_resolve_sym (char *name, g95_namespace * ns, void **p);

int g95x_common_valid (g95_common_head *);

#define g95x_valid_namespace( ns ) \
  ( (ns)->proc_name || ( (ns)->state == COMP_BLOCK_DATA ) )

#define g95x_symbol_intrinsic( symbol ) \
  ( ( (symbol)->module && ( ! strcmp( (symbol)->module, "(intrinsic)" ) ) ) \
  || (symbol)->attr.intrinsic )

int g95x_symbol_valid (g95_symbol *);
int g95x_symbol_defined (g95_symbol *);
int g95x_symbol_global (g95_symbol *);

const char *g95x_symbol_gname (g95_symbol * sym);

const char *g95x_form_type (g95_source_form);
const char *g95x_namespace_type (g95_namespace *);
const char *g95x_formal_arg_type (g95_formal_arglist *);
const char *g95x_expr_type (g95_expr *);
const char *g95x_iop_type (g95_intrinsic_op);
const char *g95x_ref_type (g95_ref *);
const char *g95x_flavor_type (g95_symbol *);
const char *g95x_get_constant_as_string (g95_expr *, int);
const char *g95x_array_spec_type (g95_array_spec *);
const char *g95x_coarray_spec_type (g95_coarray_spec *);
const char *g95x_statement_type (g95x_statement *);
const char *g95x_interface_type (interface_type);
const void *g95x_type_id (g95_typespec *);
const char *g95x_type_name (g95_typespec * ts);

void g95x_get_s_acv (g95_symbol *,
		     g95_array_spec **, g95_coarray_spec **, g95_expr **);

void g95x_get_lk (g95_typespec *, g95_expr **, g95_expr **);

void g95x_get_ac (g95_array_spec *, g95_coarray_spec *,
		  g95_array_spec **, g95_coarray_spec **);

void g95x_get_c_lkacv (g95_component *, g95_expr **, g95_expr **,
		       g95_array_spec **, g95_coarray_spec **, g95_expr **);

void g95x_get_s_lkacv (g95_symbol *, g95_expr **, g95_expr **,
		       g95_array_spec **, g95_coarray_spec **, g95_expr **);



/* xml-bbt.c */

typedef struct g95x_bbt_id
{
  BBT_HEADER (g95x_bbt_id);
  int defined;
  unsigned hidden:1, referenced:1;
  g95x_pointer p;
  char *name;
  g95x_pointer q;
} g95x_bbt_id;

typedef struct g95x_bbt_id_root
{
  g95x_bbt_id *root;
  int bbt_id_number;
} g95x_bbt_id_root;




g95x_bbt_id_root root_bbt_id;

g95x_bbt_id *g95x_new_bbt_id (g95x_pointer, const char *, g95x_bbt_id_root *);
g95x_bbt_id *g95x_new_bbt_id_an (g95x_bbt_id_root *);
g95x_bbt_id *g95x_get_bbt_id (g95x_pointer, const char *, g95x_bbt_id_root *);
void g95x_free_bbt_ids (g95x_bbt_id_root *);
void g95x_check_bbt_ids ();

/* xml-object.c */

FILE *g95x_out;

char * g95x_f2xml (const char *);

const char* g95x_current_obj_type ();
const char* g95x_current_obj_name ();

const char* g95x_get_xml_file ();

const char* g95x_get_f95_file ();

g95x_pointer g95x_get_obj_id_an ();

int g95x_add_inc ();
int g95x_end_inc ();

int g95x_add_fle (g95_file *);
int g95x_end_fle ();

int g95x_add_als1 (const char *, const char *, const char *, const void *);
int g95x_add_obj1 (const char *, const char *, int, const void *);
int g95x_end_obj1 ();

int g95x_add_att1_str (const char *, const char *);
int g95x_add_att1_cst (const char *, const char *);
int g95x_add_att1_int (const char *, const int);
int g95x_add_att1_obj (const char *, const void *);
int g95x_add_att1_als (const char *, const char *, const void *);
int g95x_add_att1_und (const char *);
int g95x_add_att1_chr (const char *, char);
int g95x_end_att1 ();

int g95x_add_lst1 (const char *);
int g95x_end_lst1 ();

int g95x_psh_lst1_obj (const void *);
int g95x_psh_lst1_chr (char c);
int g95x_psh_lst1_int (int);
int g95x_psh_lst1_und ();
int g95x_psh_lst1_str (const char *);
int g95x_psh_lst1_als (const char *, const void *);
int g95x_add_sct1 (const char *);
int g95x_end_sct1 ();

extern int g95x_put_txt1_cd;
int g95x_add_txt1 ();
int g95x_put_txt1 (const char *, const char *, int, int, int *);
int g95x_end_txt1 (int *);





int g95x_exists_obj_id (g95x_pointer);
int g95x_exists_als_id (const char *, g95x_pointer);
int g95x_defined_obj_id (g95x_pointer);
int g95x_defined_als_id (const char *, g95x_pointer);
g95x_voidp_list *g95x_set_voidp_prev (g95x_voidp_list * f);

int g95x_add_dmp ();
int g95x_end_dmp ();


/* xml-locate.c */

char g95x_char_move (g95x_char_index *, int, int);
void g95x_locate_expr_nokind (g95x_locus *, g95_expr *);
void g95x_locate_name (g95x_locus *);
char *g95x_get_code_substr (g95x_locus *, char *, int);
char *g95x_get_name_substr (g95x_locus *, char *);
void g95x_refine_location (g95x_locus *, g95x_locus **, int *);
int g95x_compare_locus (const g95x_locus *, const g95x_locus *);
int g95x_iok (g95x_locus *, const char *, g95x_locus *);
int g95x_opl (const g95x_locus *, g95x_locus *, const char *, const char *);
int g95x_locate_attr (g95x_statement *, const char *, g95x_locus *);
int g95x_match_word (g95x_locus *, const char *);
int g95x_parse_letter_spec_list (g95x_locus *, int *, g95x_letter_spec *);
void g95x_refine_location1 (g95x_locus *, g95_linebuf **, int *);
void g95x_refine_location2 (g95x_locus *, g95_linebuf **, int *);

/* xml-dump-source.c */

extern int g95x_dump_source_enabled;
void g95x_dump_source_code (g95x_locus *, int, int);
int g95x_close_file (g95_file *);
void g95x_xml_escape (char *, const char *, int);

/* xml-dump.c */

#define sline_(x) #x
#define sline(x) sline_(x)
#define _S(x) (  \
  g95x_option.xml_watch \
    ? g95x_watch ("<string\0><![CDATA[\0" x "\0" "]]><loc f=\"" __FILE__ "\" l=\"" sline(__LINE__) "\"/></string>")  \
    : x \
)

void g95x_dump (g95_namespace *);

/* xml-example.c */

int g95x_xml_example_switch (const char *);

int g95x_xml_example (const char *);
const char * g95x_watch (const char *);
int g95x_add_watch ();
int g95x_end_watch ();
int g95x_get_watch_id ();
int g95x_new_watch_id ();





