
#define G95X_COMPONENT \
struct {                      \
  struct g95_symbol* derived; \
} x

#define G95X_ARRAY_REF                            \
struct {                                          \
    g95x_locus section_where[G95_MAX_DIMENSIONS]; \
} x


#define G95X_FILE \
struct {                         \
    g95_source_form form;        \
    struct g95x_cpp_line *cl;    \
    void *input;                 \
    int ext_locs_index;          \
    int width;                   \
    char* text;                  \
    char* cur;                   \
    int linenum;                 \
    int linetot;                 \
    int include_level;           \
    int closed;                  \
} x


#define G95X_CHARLEN \
struct {                       \
    struct g95_charlen *alias; \
    g95x_locus chst_where;     \
    int star1;                 \
    int star;                  \
} x


#define G95X_TYPESPEC \
struct {                       \
    struct g95x_kind *kind;    \
    int dble;                  \
    int star1;                 \
    g95x_locus where;          \
    g95x_locus where_derived;  \
    g95x_locus where_interface;\
} x

#define G95X_ARRAY_SPEC \
struct {                                             \
    struct g95_array_spec *alias;                    \
    g95x_locus where;                                \
    g95x_locus assumed_where;                        \
    g95x_locus shape_spec_where[G95_MAX_DIMENSIONS]; \
} x


#define G95X_COARRAY_SPEC \
struct {                                             \
    struct g95_coarray_spec *alias;                  \
    g95x_locus where;                                \
    g95x_locus assumed_where;                        \
    g95x_locus shape_spec_where[G95_MAX_DIMENSIONS]; \
} x



#define G95X_FORMAL_ARGLIST \
struct {                       \
  g95x_locus where;            \
} x



#define G95X_INTERFACE \
struct {                        \
    struct g95_interface* head; \
    g95x_intf_info* intf_info;  \
} x



#define G95X_ST_LABEL \
struct {                       \
    struct g95x_locus where;   \
} x


#define G95X_USER_OP \
struct {                  \
    char* name, *module;  \
} x


#define G95X_SYMBOL \
struct {                                     \
    int imported;                            \
    int is_generic;                          \
    int visible;                             \
    int use_assoc;                           \
    struct g95_expr *bind;                   \
    struct g95_intrinsic_sym* isym;          \
    g95x_locus where, end_where;             \
    char* module_name;                       \
} x


#define G95X_NAMESPACE \
struct {                                                    \
  g95x_kind *k_list, *k0;                                   \
  struct g95x_statement *statement_head, *statement_tail;   \
  struct g95_data* data1;                                   \
  struct g95_equiv* equiv1;                                 \
  g95x_locus where;                                         \
} x


#define G95X_REF \
struct {                   \
  g95_typespec* ts;        \
  g95x_locus where;        \
  g95x_locus where_obj;    \
} x


#define G95X_INTRINSIC_SYM \
struct {                       \
    int used;                  \
} x


#define G95X_EXPR \
struct {                          \
    struct g95_symtree *generic;  \
    struct g95_expr *full;        \
    struct g95_symbol *symbol;    \
    struct g95x_locus where;      \
    int has_ts;                   \
    int is_operator;              \
    g95_intrinsic_op op;          \
    g95_symtree *uop;             \
    g95_interface *iop;           \
    char* boz;                    \
    g95x_locus where_op;          \
    g95x_locus where_sy;          \
} x



#define G95X_EQUIV \
struct {                               \
    int imported;                      \
    struct g95_equiv *prev;            \
    g95x_locus where_equiv;            \
} x


#define G95X_CLOSE \
struct {                \
  g95x_locus err_where; \
} x


#define G95X_FILEPOS \
struct {                  \
  g95x_locus err_where;   \
} x


#define G95X_DT \
struct {                             \
  g95x_locus err_where, end_where,   \
  eor_where, format_label_where,     \
  io_unit_where, format_expr_where,  \
  namelist_where;                    \
} x




#define G95X_INQUIRE \
struct {                 \
  g95x_locus err_where;  \
} x


#define G95X_WAIT \
struct {                                       \
  g95x_locus err_where, end_where, eor_where;  \
} x



#define G95X_OPEN \
struct {                 \
  g95x_locus err_where;  \
} x

#define G95X_FLUSH \
struct {                 \
  g95x_locus err_where;  \
} x

#define G95X_CODE \
struct {                               \
    int is_print;                      \
    struct g95_symtree *generic;       \
    struct g95x_statement *statement;  \
    struct g95_symbol *symbol;         \
    g95_interface *intr_op;            \
    g95x_locus where;                  \
    g95_statement sta;                 \
    g95x_locus label_where,            \
      label2_where, label3_where,      \
      call_where, assign_where;        \
} x


#define G95X_DATA_VARIABLE \
struct {                      \
    struct g95x_locus where;  \
} x

#define G95X_DATA_VALUE \
struct {                 \
  g95_expr *repeat;      \
  g95x_locus where;      \
} x


#define G95X_DATA \
struct {                     \
    struct g95_data *prev;   \
    g95x_locus where;        \
} x

#define G95X_CONSTRUCTOR \
struct {                        \
    int dummy;                  \
    char* name;                 \
    g95x_locus where_kw;        \
    g95x_locus where;           \
    struct g95_component* comp; \
    struct g95_symbol* type;    \
} x


#define G95X_USE_RENAME \
struct {                                        \
  g95x_ext_locus *local_name_el, *use_name_el;  \
  int mark_el;                                  \
  g95x_locus where, local_where, use_where;     \
  interface_type type;                          \
  int done;                                     \
  union {                                       \
    g95_symtree *st_gnrc;                       \
    g95_symbol *sym;                            \
    g95_user_op *uop;                           \
    g95_interface *iop_intf;                    \
  } u_local;                                    \
  union {                                       \
    g95_symbol *sym;                            \
    g95_interface *gnrc;                        \
    g95_interface *uop;                         \
    g95_interface *iop_intf;                    \
  } u_use;                                      \
} x

#define G95X_ACTUAL_ARGLIST \
struct {                          \
  g95x_locus name_where;          \
  g95x_locus label_where;         \
  g95x_locus where;               \
} x

#define G95X_ALLOC  \
struct {            \
  g95x_locus where; \
  int dealloc;      \
} x

#define G95X_ITERATOR \
struct {              \
  g95x_locus where;   \
} x

#define G95X_FORALL_ITERATOR \
struct {                     \
  g95x_locus where;          \
} x


