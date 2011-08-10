
typedef struct g95x_kind
{
  struct g95_expr *kind;
  struct g95x_kind *next;
} g95x_kind;

/* g95x definitions */


typedef struct g95x_symbol_list
{
  struct g95_symbol *symbol;
  struct g95x_symbol_list *next;
} g95x_symbol_list;

#define g95x_get_symbol_list() g95_getmem( sizeof( g95x_symbol_list ) )


typedef enum
{
  G95X_LOCUS_NAME = 0x00000001,	/* symbol, generic    */
  G95X_LOCUS_KWRD = 0x00000004,	/* call keyword       */
  G95X_LOCUS_MMBR = 0x00000008,	/* structure member   */
  G95X_LOCUS_ASSC = 0x0000000a,	/* use association    */
  G95X_LOCUS_CMMN = 0x00000010,	/* common             */
  G95X_LOCUS_LABL = 0x00000020,	/* label              */
  G95X_LOCUS_USOP = 0x00000040,	/* user op            */
  G95X_LOCUS_INOP = 0x000000a0,	/* intrinsic op       */
  G95X_LOCUS_BOZZ = 0x00000100,	/* boz constant       */
  G95X_LOCUS_STRG = 0x00000200,	/* string constant    */
  G95X_LOCUS_HLLR = 0x00000400,	/* hollerith constant */
  G95X_LOCUS_LANG = 0x00000a00	/* fortran language   */
} g95x_locus_type;

typedef struct g95x_ext_locus
{
  g95x_locus loc;
  g95x_locus_type type;
} g95x_ext_locus;

typedef struct g95x_voidp_list
{
  enum
  {
    G95X_VOIDP_UNKNOWN,
    G95X_VOIDP_GENERIC,
    G95X_VOIDP_SYMBOL,
    G95X_VOIDP_USEROP,
    G95X_VOIDP_COMMON,
    G95X_VOIDP_INTROP,
    G95X_VOIDP_INTROP_IDX,
    G95X_VOIDP_COMPNT
  } type;
  union
  {
    struct g95_symtree *generic;
    struct g95_symbol *symbol;
    struct g95_symtree *userop;
    struct g95_common_head *common;
    struct g95_interface *introp;
    struct g95_component *component;
    void *voidp;
    int introp_idx;
  } u;
  int dimension;
  int init;
  int length;
  int in_derived;
  struct g95x_voidp_list *next;
  struct g95x_voidp_list *prev;
  g95x_locus where;
  g95x_locus where_obj;
  g95x_locus where_all;
} g95x_voidp_list;

#define g95x_get_voidp_list() g95_getmem( sizeof( g95x_voidp_list ) )

typedef struct g95x_implicit_spec
{
  struct g95x_implicit_spec *next;
  g95x_locus where;
  g95_typespec ts;
} g95x_implicit_spec;

#define g95_get_implicit_spec() (g95x_implicit_spec*)g95_getmem( sizeof( g95x_implicit_spec ) )



typedef struct g95x_statement
{
  g95_statement type;
  g95x_locus where, where_block, where_construct;
  struct g95_code *code;
  struct g95x_statement *a_next, *a_prev;	/* Doubly linked list, all statements 
						   chained together 
						 */
  struct g95x_statement *next, *prev;	/* Doubly linked list, per namespace */
  struct g95x_statement *eblock;	/* Points to ELSE/ENDIF for IF statements 
					   CASE/END SELECT for SELECT CASE, etc...
					 */
  struct g95x_statement *fblock;	/* sta->fblock->eblock == sta */


  struct g95_st_label *here;

  struct g95_symbol *block;
  union
  {
    struct
    {
      g95x_symbol_list *symbol_head, *symbol_tail;
      struct g95x_statement *st_interface;
      struct
      {
	interface_type type;
	struct g95_namespace *ns;
	struct g95_user_op *uop;
	struct g95_symtree *generic;
	int op;
      } info;
    } interface;
    struct
    {
      struct g95_equiv *e1;
      struct g95_equiv *e2;
    } equiv;
    struct
    {
      struct g95_data *d1;
      struct g95_data *d2;
    } data;
    struct
    {
      struct g95_st_label *here;
    } format;
    struct
    {
      g95_typespec ts;
      struct g95_array_spec *as;
      struct g95_coarray_spec *cas;
      symbol_attribute attr;
      g95x_voidp_list *decl_list;
      struct g95_expr *bind;
    } data_decl;
    struct
    {
      g95x_voidp_list *decl_list;
    } import;
    struct
    {
      g95x_voidp_list *decl_list;
    } modproc;
    struct
    {
      g95x_voidp_list *common_list;
    } common;
    struct
    {
      g95x_voidp_list *enumerator_list;
    } enumerator;
    struct
    {
      int only;
      struct g95_use_rename *rename_list;
      g95x_voidp_list module;
    } use;
    struct
    {
      int n;
    } nullify;
    struct
    {
      symbol_attribute attr;
      g95x_locus where;
    } subroutine;
    struct
    {
      symbol_attribute attr;
      g95_typespec ts;
      g95x_locus where;
    } function;
    struct
    {
      g95x_implicit_spec *list_is;
    } implicit;
    struct
    {
      g95x_voidp_list *namelist_list;
    } namelist;
  } u;

  g95x_ext_locus *ext_locs;
  int n_ext_locs;

  int enum_bindc;
  int in_derived;
  int implied_enddo;
  struct g95_namespace *ns;

} g95x_statement;


typedef enum
{
  G95X_CHAR_NONE = 0x00,
  G95X_CHAR_CODE = 0x01,
  G95X_CHAR_COMM = 0x02,
  G95X_CHAR_CTFX = 0x04,
  G95X_CHAR_CTFR = 0x08,
  G95X_CHAR_ZERO = 0x10,
  G95X_CHAR_STRG = 0x20,
  G95X_CHAR_SMCL = 0x40,
  G95X_CHAR_RIMA = 0x80
} g95x_char_type;

#define g95x_init_callback( xtc ) memset( xtc, 0, sizeof( g95x_traverse_callback ) )


typedef struct g95x_option_t
{
  int enable;
  int argc;
  char **argv;
  char *xmlns;
  char *xmlns_uri;
  char *stylesheet;
  char *stylesheet_type;
  int xml_type;
  int xml_no_header;
  int xml_defn;
  int xml_xul;
  int xml_sc;
  int xml_no_ns;
  int xml_example;
  int xml_watch;
  int xml_dump_mask;
} g95x_option_t;

extern g95x_option_t g95x_option;


typedef struct g95x_char_index
{
  int column;
  g95_linebuf *lb;
} g95x_char_index;

typedef struct g95x_intf_info
{
  interface_type type;
  union
  {
    char *generic_name;
    char *uop_name;
    int iop_idx;
  } u;
  char *module;
  struct g95_interface *intf;
  struct g95x_intf_info *next;
} g95x_intf_info;

#define g95x_get_intf_info() (g95x_intf_info*)g95_getmem( sizeof( g95x_intf_info ) )

typedef struct g95x_cpp_mask_item
{
  short c;
  short c1;
  short c2;
  short done;
  g95x_pointer id;
} g95x_cpp_mask_item;

typedef struct g95x_cpp_mask
{
  int norig, ncode;
  short *orig;
  g95x_cpp_mask_item *code;
  short data[0];
} g95x_cpp_mask;
