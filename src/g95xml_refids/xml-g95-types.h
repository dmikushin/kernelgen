typedef struct g95x_locus {
    struct g95_linebuf *lb1;
    int column1;
    struct g95_linebuf *lb2;
    int column2;
} g95x_locus;


typedef struct g95x_kind {
    struct g95_expr *kind;
    struct g95x_kind *next;
} g95x_kind;

/* g95x definitions */


typedef struct g95x_symbol_list {
    struct g95_symbol *symbol;
    struct g95x_symbol_list *next;
} g95x_symbol_list;

#define g95x_get_symbol_list() g95_getmem( sizeof( g95x_symbol_list ) )


typedef struct g95x_voidp_list {
    enum {
        G95X_VOIDP_UNKNOWN,
        G95X_VOIDP_GENERIC,
        G95X_VOIDP_SYMBOL,
        G95X_VOIDP_USEROP,
        G95X_VOIDP_COMMON,
        G95X_VOIDP_INTROP,
        G95X_VOIDP_INTROP_IDX,
        G95X_VOIDP_COMPNT 
    } type;  
    union {
        struct g95_symtree* generic;
        struct g95_symbol* symbol;
        struct g95_symtree *userop;
        struct g95_common_head *common;
        struct g95_interface *introp;
        struct g95_component* component;
        void *voidp;
        int introp_idx;
    } u;
    int dimension;
    int init;
    struct g95x_voidp_list *next;
    struct g95x_voidp_list *prev;
} g95x_voidp_list;

#define g95x_get_voidp_list() g95_getmem( sizeof( g95x_voidp_list ) )

typedef enum {
    G95X_LOCUS_NAME =  0, /* symbol, generic    */
    G95X_LOCUS_SMBL =  1,
    G95X_LOCUS_KWRD =  2, /* call keyword       */
    G95X_LOCUS_MMBR =  3, /* structure member   */
    G95X_LOCUS_ASSC =  4, /* use association    */
    G95X_LOCUS_CMMN =  5, /* common             */
    G95X_LOCUS_LABL =  6, /* label              */
    G95X_LOCUS_USOP =  7, /* user op            */
    G95X_LOCUS_GNRC =  8, /* generic            */
    G95X_LOCUS_INOP =  9, /* intrinsic op       */
    G95X_LOCUS_BOZZ = 10, /* boz constant       */
    G95X_LOCUS_STRG = 11, /* string constant    */
    G95X_LOCUS_HLLR = 12, /* hollerith constant */
    G95X_LOCUS_INTR = 13, /* intrinsic          */
    G95X_LOCUS_LANG = 14, /* fortran language   */
    G95X_LOCUS_SIZE = 15
} g95x_locus_type;

typedef struct g95x_ext_locus {
    g95x_locus loc;
    g95x_locus_type type;

    struct g95_expr* fct_expr;        /* when matching a function call ( generic resolution ) */
    struct g95_code* sub_code;        /* when matching a call statement ( generic resolution ) */
    struct g95_code* ass_code;        /* assignment code */

    union {
        struct {
            struct g95_symbol *derived_type; 
            struct g95_component *component;
        } mmbr;
        struct {
            struct g95_st_label* label;     
        } labl;
        struct {
            struct g95_expr* expr;
            struct g95_symtree *uop;     
            struct g95_symbol* symbol;
            struct g95_interface* intf;
        } usop;
        struct {
            struct g95_common_head *common;
        } cmmn;
        struct {
        } kwrd;
        struct {
            struct g95_expr* expr;       
            struct g95_interface* intf;
            struct g95_symbol* symbol;
            g95_intrinsic_op iop;
        } inop;
        struct {
            struct g95_interface *intf;
            struct g95_symbol* symbol;
        } gnrc;
        struct {
            struct g95_symbol* symbol;
        } smbl;
        struct {
            struct g95_symbol* symbol;
            struct g95_interface* intf;
        } assc;
        struct {
            struct g95_intrinsic_sym* isym;
        } intr;
    } u;
  
} g95x_ext_locus; 


typedef struct g95x_statement {
    g95_statement type;
    g95x_locus where;
    struct g95_code *code;
    struct g95x_statement *a_next, *a_prev; /* Doubly linked list, all statements 
                                               chained together 
                                             */
    struct g95x_statement *next, *prev; /* Doubly linked list, per namespace */
    struct g95x_statement *eblock;      /* Points to ELSE/ENDIF for IF statements 
                                           CASE/END SELECT for SELECT CASE, etc...
				         */
    struct g95x_statement *fblock;      
                                      
				      
    struct g95_st_label *here;
  
    struct g95_symbol *block; 
    union {
        struct {
            interface_type type;
            g95x_symbol_list *symbol_head, *symbol_tail;
            struct g95x_statement *st_interface;
        } interface;
        struct {
            struct g95_equiv *e1;
            struct g95_equiv *e2;
        } equiv;
        struct {
            struct g95_data *d1;
            struct g95_data *d2;
        } data;
        int is_print;
        struct {
            struct g95_st_label *here; 
        } format;
        struct {
            g95_typespec ts;
            struct g95_array_spec *as;
            struct g95_coarray_spec *cas;
            symbol_attribute attr;
            g95x_voidp_list *decl_list;
        } data_decl;
        struct {
            g95x_voidp_list *common_list;
        } common;
        struct {
            g95x_voidp_list *enumerator_list;
        } enumerator;
        struct {
            int only;
        } use;
        struct {
            int n;
        } nullify;
        struct {
            symbol_attribute attr;
        } subroutine;
        struct {
            symbol_attribute attr;
            g95_typespec ts;
        } function;
        struct {
            g95_typespec ts[G95_LETTERS];
        } implicit;
    } u;
    
    g95x_ext_locus *ext_locs;
    int n_ext_locs;
    
    int enum_bindc;
    int in_derived;
    int implied_enddo;
    struct g95_namespace *ns;

} g95x_statement;


typedef enum {
  G95X_CHAR_NONE = ' ',
  G95X_CHAR_CODE = 'C',
  G95X_CHAR_COMM = '!',
  G95X_CHAR_CTFX = '>',
  G95X_CHAR_CTFR = '&',
  G95X_CHAR_ZERO = '0',
  G95X_CHAR_STRG = 'S'
} g95x_char_type;

struct g95_expr;
struct g95_constructor;
struct g95_symbol;
struct g95_ref;
struct g95_component;
struct g95_symtree;
struct g95_data_variable;
struct g95_code;

typedef struct g95x_traverse_callback {
  int (*expr_callback)( struct g95_expr*, void* );
  void *expr_data;
  int (*constructor_callback)( struct g95_constructor*, void* );
  void *constructor_data;
  int (*symbol_callback)( struct g95_symbol*, void* );
  void *symbol_data;
  int (*ref_callback)( struct g95_ref*, void* );
  void *ref_data;
  int (*code_callback)( struct g95x_statement*, void* );
  void *code_data;
  int (*component_callback)( struct g95_component*, void* );
  void *component_data;
  int (*common_callback)( struct g95_symtree*, void* );
  void *common_data;
  int (*data_var_callback)( struct g95_data_variable*, void* );
  void *data_var_data;
  int (*io_var_callback)( struct g95_code*, void* );
  void *io_var_data;
} g95x_traverse_callback;

#define g95x_init_callback( xtc ) memset( xtc, 0, sizeof( g95x_traverse_callback ) )


typedef struct g95x_option_t {
  FILE *g95x_out;
  int reduced_exprs;
  int canonic;
  int check;
  int enable;
  int code_only;
  int intrinsics;
  int ext_locs_in_statement;
  char *file;
  int indent;
  int argc;
  char **argv;
} g95x_option_t;

extern g95x_option_t g95x_option;
