
/* decl.c */

g95_typespec* g95x_get_current_ts();
symbol_attribute* g95x_get_current_attr();
g95x_voidp_list* g95x_take_decl_list();
g95_array_spec* g95x_take_current_as();


/* symbol.c */

g95_typespec* g95x_get_implicit_ts();


/* intrinsic.c */

void g95x_get_intrinsics_info( 
  g95_intrinsic_sym**, g95_intrinsic_sym**, int*, int*
);

/* scanner.c */

void g95x_print_mask();
void g95x_print_files();

int g95x_current_line( g95_file* );

g95_file* g95x_get_file_head();

/* arith.c */

extern int g95x_simplify_state[4096];    
extern int g95x_simplify_index;

#define g95x_simplify_none() g95x_simplify_state[g95x_simplify_index]
#define g95x_simplify_push( x ) \
  do { g95x_simplify_state[++g95x_simplify_index] = x; } while( 0 )
#define g95x_simplify_pop() --g95x_simplify_index
#define g95x_simplify_init() g95x_simplify_push( 1 )

/* match.c */

int g95x_enum_bindc();
g95x_voidp_list* g95x_take_common_list();
g95_array_spec* g95x_take_current_as();
g95_coarray_spec* g95x_take_current_cas();
g95x_voidp_list* g95x_take_enumerator_list();


/* expr.c */

g95x_kind *g95x_get_kind(g95_namespace *);

/* module.c */

int g95x_get_only_flag();

/* xml-basic.c */

int g95x_parse_arg( int, char *argv[], int );

int g95x_nodummy_expr( g95_expr * );
try g95x_simplify_expr( g95_expr **expr );
try g95x_simplify_spec_expr(g95_expr **e);

void g95x_push_statement( g95_statement, g95_code*, g95x_locus* );

void g95x_push_locus( g95x_locus*, g95x_locus_type );
void g95x_push_locus_mmbr( g95x_locus*, g95_symbol *derived_type );
void g95x_push_locus_labl( g95x_locus *loc, g95_st_label *label );
void g95x_push_locus_usop( g95x_locus *loc, g95_expr *e );
void g95x_push_locus_inop( g95x_locus *loc, g95_intrinsic_op op );

g95x_ext_locus* g95x_get_current_ext_locus_ref();

g95x_locus g95x_pop_locus();
void g95x_mark_1( g95x_locus* );
void g95x_mark_2( g95x_locus* );
int g95x_locus_valid( g95x_locus *xw );

void g95x_set_kwrd_locus_expr( g95_expr* e );
void g95x_set_cmmn_locus_common( g95_common_head *c );


int g95x_init();
int g95x_done();

void g95x_error( int, int, const char*, const char*, ... );

void g95x_set_io_code( g95_code * );
void g95x_set_iolength_code( g95_code * );
void g95x_set_nullify_code( g95_code * );
void g95x_set_is_print_stmt( int );

char* g95x_f2xml( const char* );

void g95x_free_namespace( g95_namespace *ns );
int g95x_symbol_global( g95_symbol*, g95_namespace* );
char* g95x_get_code_substring( char *string, g95x_locus *xw, int start, int length, int name );

void g95x_fixup_generics( g95_namespace * );


extern g95x_statement *g95x_statement_head, 
                      *g95x_statement_tail;
                      
extern const char* g95x_current_intrinsic_module_name;                      
extern char** g95x_builtin_macros;


/* xml-traverse.c */

void g95x_traverse_symtree( g95_symtree *, void (*)( g95_symtree*, void* ), void* );
void g95x_traverse_expr( g95_expr*, g95x_traverse_callback* );      
void g95x_traverse_symbol( g95_symbol*, g95x_traverse_callback* );
void g95x_traverse_code( g95x_statement*, g95x_traverse_callback* );
void g95x_traverse_data_var( g95_data_variable*, g95x_traverse_callback* );
void g95x_traverse_io_var( g95_code*, g95x_traverse_callback* );
void g95x_traverse_array_spec( g95_array_spec *as, g95x_traverse_callback *xtc );

/* xml-print.c */

const char* REFFMT0;
const char* REFFMT1;
#ifdef G95X_64BITS
typedef unsigned long long g95x_pointer;
#else
typedef unsigned int g95x_pointer;
#endif


void g95x_print_namespace( g95_namespace* );
int g95x_print( const char *, ... );
void g95x_print_string( const char*, int, int );
void g95x_print_intrinsics();
void g95x_print_statements();
int g95x_start( g95_namespace* );
int g95x_end();

int g95x_print_statement_ext_locs( g95x_statement* sta );
int g95x_number_statement_ext_locs( g95x_statement* sta );


#define g95x_warn( ... ) \
    g95x_error( 0, __LINE__, __FILE__, __VA_ARGS__ )       

#define g95x_die( ... ) \
    g95x_error( 1, __LINE__, __FILE__, __VA_ARGS__ )   

void g95x_resolve_gen( char* name, g95_namespace *ns, void** p );
void g95x_resolve_uop( char* name, g95_namespace *ns, void** p );
void g95x_resolve_iop( g95_intrinsic_op type, g95_namespace *ns, void** p );
void g95x_resolve_sym( char* name, g95_namespace *ns, void** p );

g95_intrinsic_sym* g95x_get_intrinsic_by_symbol( g95_symbol *sym );
int g95x_symbol_valid( g95_symbol* sym );


/* xml-bbt.c */

typedef struct g95x_bbt_id {
    BBT_HEADER( g95x_bbt_id );
    int defined;
    g95x_pointer p;
    char *name;
    g95x_pointer q;
} g95x_bbt_id;

typedef struct g95x_bbt_id_root {
    g95x_bbt_id* root;
    int bbt_id_number;
} g95x_bbt_id_root;


typedef struct bbt_list_item_data {
    g95_namespace *ns;
    int printed;
    g95x_bbt_id_root bbt_root;
} bbt_list_item_data;


g95x_bbt_id_root root_bbt_id;

g95x_bbt_id *g95x_new_bbt_id( g95x_pointer p, char* name, g95x_bbt_id_root* bbt_root );
g95x_bbt_id *g95x_get_bbt_id( g95x_pointer p, char* name, g95x_bbt_id_root* bbt_root );
void g95x_free_bbt_ids( g95x_bbt_id_root *bbt_root );
void g95x_check_bbt_ids();



