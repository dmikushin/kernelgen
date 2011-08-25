#include "g95.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

static char* char_null = NULL;
char** g95x_builtin_macros = &char_null;



g95x_option_t g95x_option = {
  NULL, 0, 0, 0, 0, 0, 0, 0, NULL, 0, 0, NULL
};

const char* g95x_current_intrinsic_module_name = NULL;


/*
 * Mutate a Fortran file name to an xml file name
 */

char* g95x_f2xml( const char* file ) {
  int length;
  char *xml;
  int i;
  
  length = strlen( file );
  xml = g95_getmem( length + 6 );
  strcpy( xml, file );
  
  for( i = length; i >= 0; i-- ) {
    if( xml[i] == '.' )
      break;
  }
  if( i == 0 ) 
    g95x_die( "Cannot derive a valid xml file name from '%s'", file );
  
  xml[i+1] = 'x';
  xml[i+2] = 'm';
  xml[i+3] = 'l';
  xml[i+4] = '\0';

  return xml;
}

int g95x_parse_arg( int argc, char *argv[], int p ) {
  if( g95x_option.argc == 0 ) {
    g95x_option.argc = argc;
    g95x_option.argv = argv;
  }
  if( ! strcmp( argv[0], "-xml-intrinsics" ) ) {
    if( p ) {
      g95x_option.intrinsics = 1;
      g95x_option.enable = 1;
    }
    return 1;
  }
  if( ! strcmp( argv[0], "-xml-ext-locs-in-statement" ) ) {
    if( p ) 
      g95x_option.ext_locs_in_statement = 1;
    return 1;
  }
  if( ! strcmp( argv[0], "-xml-canonic" ) ) {
    if( p ) 
      g95x_option.canonic = 1;
    return 1;
  }
  if( ! strcmp( argv[0], "-xml-check" ) ) {
    if( p ) 
      g95x_option.check = 1;
    return 1;
  }
  if( ! strcmp( argv[0], "-c" ) ) 
    return 1;
  if( ! strcmp( argv[0], "-xml-reduced-exprs" ) ) {
    if( p ) 
      g95x_option.reduced_exprs = 1;
    return 1;
  }
  if( ! strcmp( argv[0], "-xml" ) ) {
    if( p )
      g95x_option.enable = 1;
    return 1;
  }
  if( ! strcmp( argv[0], "-xml-out=-" ) ) {
    if( p ) {
      g95x_option.enable = 1;
      g95x_option.g95x_out = stdout;
    }
    return 1;
  }
  if( ! strcmp( argv[0], "-xml-indent" ) ) {
    if( p )
      g95x_option.indent = 1;
    return 1;
  }
  if( ! strcmp( argv[0], "-xml-code-only" ) ) {
    if( p )
      g95x_option.code_only = 1;
    return 1;
  }
  if( ! strncmp( argv[0], "-xml-out=", 9 ) ) {
    if( p ) {
      g95x_option.enable = 1;
      g95x_option.file = g95x_f2xml( argv[0] + 9 );
    }
    return 1;
  }
  if( ! strncmp( argv[0], "-xml-macros=", 12 ) ) {
    if( ! p ) 
      return 1;

/*
char *builtin_macros[] = {
   "__G95__ 0",
   "__G95_MINOR__ 50",
   "__FORTRAN__ 95",
   "__GNUC__ 4",
   NULL
};
*/


    char* macros = argv[0] + 12;
    FILE *fp = fopen( macros, "r" );
    int nlines = 0;
    int sizemax = 0;
    char buffer256[256];
    int i;
    int size, nline;
    int nread;
    
    if( fp == NULL )
      g95x_die( "Cannot open macros definition: `%s'\n", macros );
    
    while( ( nread = fread( buffer256, 1, 256, fp ) ) ) {
      size = 0;
      for( i = 0; i < nread; i++ ) {
        if( buffer256[i] == '\n' ) {
          nlines++;
          if( size > sizemax )
            sizemax = size;
          size = 0;
        } else {
          size++;
        }
      }
    }

    g95x_builtin_macros = g95_getmem( sizeof( char* ) * nlines );
    
    rewind( fp );
    
    char* buffer = g95_getmem( ( sizemax + 1 ) * sizeof( char ) );
    nline = 0;
    while( ( nread = fread( buffer256, 1, 256, fp ) ) ) {
      size = 0;
      for( i = 0; i < nread; i++ ) {
        if( buffer256[i] == '\n' ) {
          buffer[size++] = '\0';
          g95x_builtin_macros[nline] = g95_getmem( size * sizeof( char ) );
          strcpy( g95x_builtin_macros[nline], buffer );
          size = 0;
          nline++;
        } else {
          buffer[size++] = buffer256[i];
        }
      }
    }
    
    
    g95_free( buffer );
    
    fclose( fp );
    
    
    return 1;
  }
  if( ! strncmp( argv[0], "-xml-arch=", 10 ) ) {
    if( p ) {
      char *arch = argv[0] + 10;
      FILE *fp = fopen( arch, "r" );
      int iintr = 0, ilogl = 0, ireal = 0;
      
      
      if( fp == NULL )
        g95x_die( "Undefined arch : `%s'\n", arch );
      
      for( ; ; ) {
        int kind, radix, digits, bit_size;
        fscanf( fp, "%d %d %d %d\n", &kind, &radix, &digits, &bit_size );
        g95_integer_kinds[iintr].kind     = kind;
        g95_integer_kinds[iintr].radix    = radix;
        g95_integer_kinds[iintr].digits   = digits;
        g95_integer_kinds[iintr].bit_size = bit_size;
        iintr++;
        if( kind == 0 )
          break;
      }
      
      for( ; ; ) {
        int kind, bit_size;
        fscanf( fp, "%d %d\n", &kind, &bit_size );
        g95_logical_kinds[ilogl].kind     = kind;
        g95_logical_kinds[ilogl].bit_size = bit_size;
        ilogl++;
        if( kind == 0 )
          break;
      }

      for( ; ; ) {
        int kind, radix, totalsize, sign_start, exp_start, 
            exp_len, exp_bias, exp_nan, man_start, man_len;
        char endian[32], msb[32];
        fscanf( 
          fp, "%d %d %s %d %d %d %d %d %d %d %d %s\n", 
          &kind, &radix, endian, &totalsize, &sign_start, &exp_start,
          &exp_len, &exp_bias, &exp_nan, &man_start, &man_len, msb
        );
        g95_real_kinds[ireal].kind       = kind;
        g95_real_kinds[ireal].radix      = radix;
        g95_real_kinds[ireal].endian     = strcmp( endian, "END_LITTLE" ) 
                                         ? END_BIG : END_LITTLE;
        g95_real_kinds[ireal].totalsize  = totalsize;
        g95_real_kinds[ireal].sign_start = sign_start;
        g95_real_kinds[ireal].exp_start  = exp_start;
        g95_real_kinds[ireal].exp_len    = exp_len;
        g95_real_kinds[ireal].exp_bias   = exp_bias;
        g95_real_kinds[ireal].exp_nan    = exp_nan;
        g95_real_kinds[ireal].man_start  = man_start;
        g95_real_kinds[ireal].man_len    = man_len;
        g95_real_kinds[ireal].msb        = strcmp( msb, "MSB_IMPLICIT" ) 
                                         ? MSB_EXPLICIT : MSB_IMPLICIT;
        ireal++;
        if( kind == 0 )
          break;
      }
      
      
      fclose( fp );
    }
    return 1;
  }
  return -1;
}

int g95x_init() {

  g95x_simplify_init();
  return 1;
}

int g95x_done() {
  if( g95x_option.g95x_out && ( g95x_option.g95x_out != stdout ) ) {
    fclose( g95x_option.g95x_out );
    if( g95x_option.file )
      g95_free( g95x_option.file );
  }
  return 1;
}

/*
typedef struct {
  int implicit_none, fixed_line_length, max_frame_size, no_underscoring,
    case_upper, leading_underscore, cpp, preprocess_only, unused_module_vars,
    pedantic, bounds_check, no_backslash, fmode, unused_vars, unset_vars,
    unused_module_procs, default_integer, deps, obsolete, endian, werror,
    short_circuit, module_access_private, dollar, static_var, symbol_len,
    verbose, intrinsic_extensions, tr15581, integer_init, integer_value,
    zero_init, unused_label, line_truncation, prec_loss, pack_derived, uninit,
    q_kind, quiet, r8, l1, no_second_underscore, sloppy_char, onetrip;

  enum { LOGICAL_INIT_NONE=0, LOGICAL_INIT_TRUE,
	 LOGICAL_INIT_FALSE } logical_init;

  enum { TRACE_NONE, TRACE_FRAME, TRACE_FULL } trace;

  enum { ROUND_NEAREST, ROUND_PLUS, ROUND_MINUS, ROUND_ZERO } round;

  enum { REAL_INIT_NONE=0, REAL_INIT_ZERO, REAL_INIT_NAN, REAL_INIT_PLUS_INF,
	 REAL_INIT_MINUS_INF } real_init;

  enum { POINTER_INIT_NONE=0, POINTER_INIT_NULL,
	 POINTER_INIT_INVALID } pointer_init;

  g95_source_form form;
  g95_directorylist *include_dirs;
  g95_nowarn *nowarn;
  char *module_dir, *intrinsics;
} g95_option_t;

static void print_options() {
  if( g95_option.implicit_none )
    g95x_print( " implicit_none=\"1\"" );
  if( g95_option.fixed_line_length != 72 )
    g95x_print( " fixed_line_length=\"%d\"", g95_option.fixed_line_length );
  if( g95_option.cpp )
    g95x_print( " cpp=\"1\"" );
  if( g95_option.fmode )
    g95x_print( " fmode=\"%d\"", g95_option.fmode );
    
}
*/





void g95x_error( int die, int line, const char* file, const char *fmt, ... ) {
  va_list ap;

  fflush( stdout );
  
  fprintf( stderr, "\n" );
  
  va_start( ap, fmt );
  vfprintf( stderr, fmt, ap );
  va_end( ap );

  fprintf( stderr, " at %s:%d\n", file, line );

  if( die )
    exit( 1 );
}


/*
 * g95x_nodummy_expr returns 1 if expr is a valid expr
 */
 
 
int g95x_nodummy_expr( g95_expr *e ) {
  if( ! e ) 
    return 0;
  if( ! e->x.where.lb1 )
    return 0;
  return 1;
}

try g95x_simplify_expr( g95_expr **expr ) {
  g95_expr *copy = g95_copy_expr( *expr );
  try t;
  g95_expr* expr1 = *expr;

  if( *expr == NULL )
    return SUCCESS;
  
  g95x_simplify_push( 0 );
  copy->x.full = expr1;
  t = g95_simplify_expr( copy );
  *expr = copy;
  g95x_simplify_pop();
  

  
  return t;
}

try g95x_simplify_spec_expr( g95_expr **expr ) {
  g95_expr *copy = g95_copy_expr( *expr );
  try t;
  g95_expr* expr1 = *expr;
  
  if( *expr == NULL )
    return SUCCESS;
  
  g95x_simplify_push( 0 );
  t = g95_simplify_spec_expr( copy );

  *expr = copy;
  copy->x.full = expr1;
  
  g95x_simplify_pop();
  
  return t;
}


/*
 * Locus functions; we use a static buffer as a FIFO to 
 * store matched locs of different types.
 *
 */

static g95x_ext_locus current_statement_locus[4096]; /* should be enough */

static int current_statement_locus_index = 0;

static int compare_locus( g95x_locus *xw1, g95x_locus *xw2 ) {
  if( xw1->lb1->linenum < xw2->lb1->linenum )
    return -1;
  if( xw1->lb1->linenum > xw2->lb1->linenum )
    return +1;
  if( xw1->column1 < xw2->column1 )
    return -1;
  if( xw1->column1 > xw2->column1 )
    return +1;
  return 0;
}

static void g95x_push_locus0( 
  g95x_locus *loc, g95x_locus_type type, 
  g95_symbol *derived_type, g95_st_label *label,
  g95_expr *e, g95_intrinsic_op op
) {
  if( ( loc->lb1->linenum == loc->lb2->linenum ) 
   && ( loc->column1 == loc->column2 ) ) 
    return;

  while( current_statement_locus_index > 0 ) {
    if( compare_locus( &current_statement_locus[current_statement_locus_index-1].loc, loc ) < 0 )
      break;
    current_statement_locus_index--;
  }
  
  memset( &current_statement_locus[current_statement_locus_index], 0, sizeof( g95x_ext_locus ) );
  
  current_statement_locus[current_statement_locus_index].loc = *loc;
  current_statement_locus[current_statement_locus_index].type = type;
  
  switch( type ) {
    case G95X_LOCUS_MMBR:
      current_statement_locus[current_statement_locus_index].u.mmbr.derived_type = derived_type;
    break;
    case G95X_LOCUS_LABL:
      current_statement_locus[current_statement_locus_index].u.labl.label = label;
    break;
    case G95X_LOCUS_USOP:
      current_statement_locus[current_statement_locus_index].u.usop.expr = e;
    break;
    case G95X_LOCUS_INOP:
      current_statement_locus[current_statement_locus_index].u.inop.iop = op;
    break;
    default:
    break;
  }
  
  current_statement_locus_index++;
}

void g95x_push_locus( g95x_locus *loc, g95x_locus_type type ) {

  g95x_push_locus0( loc, type, NULL, NULL, NULL, INTRINSIC_NONE );
  
}

void g95x_push_locus_mmbr( g95x_locus *loc, g95_symbol *derived_type ) {

  g95x_push_locus0( loc, G95X_LOCUS_MMBR, derived_type, NULL, NULL, INTRINSIC_NONE );

}

void g95x_push_locus_labl( g95x_locus *loc, g95_st_label *label ) {

  g95x_push_locus0( loc, G95X_LOCUS_LABL, NULL, label, NULL, INTRINSIC_NONE );

}

void g95x_push_locus_usop( g95x_locus *loc, g95_expr *e ) {

  g95x_push_locus0( loc, G95X_LOCUS_NAME, NULL, NULL, e, INTRINSIC_NONE );

}


void g95x_push_locus_inop( g95x_locus *loc, g95_intrinsic_op op ) {

  g95x_push_locus0( loc, G95X_LOCUS_INOP, NULL, NULL, NULL, op );

}

void g95x_set_kwrd_locus_expr( g95_expr* e ) {
  int i;
  for( i = 0; i < current_statement_locus_index; i++ ) 
    if( ( current_statement_locus[i].type == G95X_LOCUS_KWRD ) 
     && ( current_statement_locus[i].fct_expr == NULL )
     && ( current_statement_locus[i].sub_code == NULL )
    )
      current_statement_locus[i].fct_expr = e;
}

void g95x_set_cmmn_locus_common( g95_common_head *c ) {
  int i;
  for( i = current_statement_locus_index - 1; i >= 0; i-- ) 
    if( ( current_statement_locus[i].type == G95X_LOCUS_CMMN ) 
     && ( current_statement_locus[i].u.cmmn.common == NULL ) 
    ){
      current_statement_locus[i].u.cmmn.common = (struct g95_common_head *)c;
      return;
    }
  g95x_die( "" );
}

g95x_locus g95x_pop_locus() {
  static const g95x_locus xwn = { NULL, -1, NULL, -1 };
  if( current_statement_locus_index > 0 ) {
    current_statement_locus_index--;
    return current_statement_locus[current_statement_locus_index].loc;
  }
  return xwn;
}

g95x_ext_locus* g95x_get_current_ext_locus_ref() {
  return current_statement_locus + ( current_statement_locus_index - 1 );
}

/*
 * Mark the beginning and end of locus
 */

void g95x_mark_1( g95x_locus *xw ) {
  xw->lb1 = g95_current_locus.lb;
  xw->column1 = g95_current_locus.column;
}
     
void g95x_mark_2( g95x_locus *xw ) {
  xw->lb2 = g95_current_locus.lb;
  xw->column2 = g95_current_locus.column;
}

int g95x_locus_valid( g95x_locus *xw ) {
  return xw->lb1 && xw->lb2;
}

/* 
   Add a statement to the list of the current namespace
   g95x_push_statement does the job, and block_push,
   block_pop, block_switch are helper functions to manage
   blocks and branchs
 */

static int statement_block_index = 0;
static g95x_statement* statement_block[4096]; /* Should be enough */
static g95x_statement sta_program;

static g95x_statement* block_push( g95x_statement* sta ) {
  statement_block[statement_block_index++] = sta;
  return sta;
}

static g95x_statement* block_last() {
  if( statement_block_index == 0 ) 
    return NULL;
  return statement_block[statement_block_index-1];
}

static g95x_statement* block_pop( g95x_statement* sta ) {
  g95x_statement* sta1 = NULL;

  if( statement_block_index > 0 ) 
    sta1 = statement_block[--statement_block_index];
  else 
    g95x_die( "Attempt to block_pop NULL\n" );

  if( sta1 )
    sta1->eblock = sta;
  
  if( ( sta->implied_enddo == 0 ) && ( sta1 != &sta_program ) )
    sta->fblock = sta1;

  return sta1; 
} 

static g95x_statement* block_switch( g95x_statement* sta ) {
  if( statement_block_index > 0 ) {
    g95x_statement* sta1 = statement_block[statement_block_index-1];
    statement_block[statement_block_index-1] = sta;
    sta1->eblock = sta;
    sta->fblock = sta1;
    return sta1;
  } else {
    g95x_die( "Attempt to block_swich to NULL\n" );
  }
  return NULL;
}   

static void block_manage( g95x_statement *sta, g95_statement type ) {
  static int count = 0;


  if( count == 0 ) {
    sta_program.type = ST_PROGRAM;
    switch( type ) {
      case ST_FUNCTION: case ST_SUBROUTINE: case ST_PROGRAM: case ST_MODULE:
      case ST_BLOCK_DATA:
      break;
      default:
        block_push( &sta_program );
    }
  }

  switch( type ) {
  
    case ST_DO: case ST_IF_BLOCK: case ST_SELECT_CASE:
    case ST_WHERE_BLOCK: case ST_FORALL_BLOCK: case ST_SUBROUTINE:
    case ST_FUNCTION: case ST_INTERFACE: case ST_PROGRAM: case ST_MODULE:
    case ST_BLOCK_DATA: case ST_DERIVED_DECL: case ST_CRITICAL:
      block_push( sta );
    break;

    case ST_ELSEIF: case ST_ELSE: 
    case ST_CASE: case ST_ELSEWHERE: case ST_CONTAINS:
      block_switch( sta );
    break;
    
    case ST_ENDDO: case ST_IMPLIED_ENDDO: case ST_ENDIF:
    case ST_END_SELECT: case ST_END_WHERE: case ST_END_TYPE:
    case ST_END_FORALL: case ST_END_SUBROUTINE:
    case ST_END_FUNCTION: case ST_END_INTERFACE:
    case ST_END_PROGRAM: case ST_END_MODULE:
    case ST_END_BLOCK_DATA: case ST_END_CRITICAL:
      block_pop( sta );
    break;
    default:
    break;
  }
  
  count++;
}


/*
 * Allocates a new statement with extra size to store packed locations
 * at the end, then transfer current locations within the statement
 * We re-initialize current_statement_index to 0
 */


static g95x_statement *get_statement() {
  return g95_getmem( sizeof( g95x_statement ) );
}

static void free_statement( g95x_statement *sta ) {
  /* we should take care of allocated interface structures */
  if( sta->type == ST_DATA_DECL ) {
    g95x_voidp_list *sl, *sl1;
    for( sl = sta->u.data_decl.decl_list; sl; ) {
      sl1 = sl->next;
      g95_free( sl );
      sl = sl1;
    }
    if( sta->u.data_decl.as )
      g95_free_array_spec( (g95_array_spec*)sta->u.data_decl.as );
    if( sta->u.data_decl.cas )
      g95_free_coarray_spec( (g95_coarray_spec*)sta->u.data_decl.cas );
  }
  if( sta->type == ST_COMMON ) {
    g95x_voidp_list *sl, *sl1;
    for( sl = sta->u.common.common_list; sl; ) {
      sl1 = sl->next;
      g95_free( sl );
      sl = sl1;
    }
  }
  if( sta->type == ST_ENUMERATOR ) {
    g95x_voidp_list *sl, *sl1;
    for( sl = sta->u.enumerator.enumerator_list; sl; ) {
      sl1 = sl->next;
      g95_free( sl );
      sl = sl1;
    }
  }
  if( sta->type == ST_PARAMETER ) {
    g95x_voidp_list *sl, *sl1;
    for( sl = sta->u.data_decl.decl_list; sl; ) {
      sl1 = sl->next;
      g95_free( sl );
      sl = sl1;
    }
  }
  if( sta->ext_locs )
    g95_free( sta->ext_locs );
  g95_free( sta ); 
}

void g95x_free_namespace( g95_namespace *ns ) {
  g95x_statement *sta;
  g95x_kind *xkind, *xkind1;
  g95x_statement *sta1;
  for( sta = ns->x.statement_head; sta; sta = sta1 ) {
    sta1 = sta->next;
    free_statement( sta );
  }


  for( xkind = ns->x.k_list; xkind; xkind = xkind1 ) {
    g95_free_expr( xkind->kind );
    xkind1 = xkind->next;
    g95_free( xkind );
  }
  ns->x.k_list = NULL;
  ns->x.statement_head = ns->x.statement_tail = NULL;
}

static int compare_locus_edge( g95x_locus *loc1, g95x_locus *loc2, int k ) {
  int l1, c1, l2, c2;
  switch( k ) {
    case 1:
      l1 = loc1->lb1->linenum; c1 = loc1->column1;
      l2 = loc2->lb1->linenum; c2 = loc2->column1;
    break;
    case 2:
      l1 = loc1->lb2->linenum; c1 = loc1->column2;
      l2 = loc2->lb2->linenum; c2 = loc2->column2;
    break;
    default:
      g95x_die( "Unexpected position in compare_locus_edge\n" );
  }
  
  if( l1 < l2 ) 
    return -1;
  if( l1 > l2 ) 
    return +1;
  if( c1 < c2 )
    return -1;
  if( c1 > c2 )
    return +1;
  
  return 0;   
}

static g95x_statement *get_statement_with_locs( g95x_locus *where ) {
  g95x_statement *sta;
  /* Set locs in current statement */
  int i, k;
  int clab = 0;
  
  for( i = 0, k = 0; i < current_statement_locus_index; i++ ) {
    if( ( compare_locus_edge( &current_statement_locus[i].loc, where, 1 ) >= 0 )
     && ( compare_locus_edge( &current_statement_locus[i].loc, where, 2 ) <= 0 )
    ) if( current_statement_locus[i].type != G95X_LOCUS_LANG )
        k++;
    /* kludge for labels */
    if( ( i == 0 ) && ( current_statement_locus[i].type == G95X_LOCUS_LABL ) ) {
      int j;
      int j0 = current_statement_locus[i].loc.column2-1;
      int j1 = where->column1-1;
      g95_linebuf *lb1 = current_statement_locus[i].loc.lb1;
      g95_linebuf *lb2 = where->lb1;
      clab = 1;
      if( lb1 == lb2 )
        for( j = j0; j < j1; j++ ) 
          clab = clab && ( lb1->mask[j] == ' ' );
      else
        clab = 0;
      if( clab ) 
        k++;
    }
  }
  
  sta = get_statement();
  sta->where = *where;

  sta->n_ext_locs = k;
  sta->ext_locs = g95_getmem( k * sizeof( g95x_ext_locus ) );

  int iel = 0;

  for( i = 0; i < current_statement_locus_index; i++ ) {
    if( ( compare_locus_edge( &current_statement_locus[i].loc, where, 1 ) >= 0 )
     && ( compare_locus_edge( &current_statement_locus[i].loc, where, 2 ) <= 0 )
    ) if( current_statement_locus[i].type != G95X_LOCUS_LANG )
        sta->ext_locs[iel++] = current_statement_locus[i];
    if( ( i == 0 ) && clab ) 
      sta->ext_locs[iel++] = current_statement_locus[i];
  }

  return sta;
} 






static int is_print_stmt = 0;

void g95x_set_is_print_stmt( int i ) {
  is_print_stmt = i;
}

/*
 * Set the right code for the statement; that's because io statements
 * span several code
 */

static void set_statement_code( g95x_statement* sta, g95_code *c ) {


  switch( sta->type ) {
    case ST_WRITE:
      if( is_print_stmt )
        sta->u.is_print = 1;
    break;
    case ST_NULLIFY: {
      g95_code *c2 = c;
      int n = 1;
      for( ; c2->next; c2 = c2->next, n++ );
      sta->u.nullify.n = n;
    }
    break;
    default:
    break;
  }
  
  sta->code = c;
  if( c ) 
    c->x.statement = sta;
    
  is_print_stmt = 0;

}


static void manage_statement_list( 
  g95x_statement **statement_head, 
  g95x_statement **statement_tail, 
  g95x_statement **sta, int ns 
) {
  if( ns ) {
    if( *statement_tail ) {
      (*statement_tail)->next = *sta;
      (*sta)->prev = (*statement_tail);
      (*statement_tail) = *sta;
    } else {
      (*statement_head) = (*statement_tail) = *sta;
    } 
  } else {
    if( *statement_tail ) {
      (*statement_tail)->a_next = *sta;
      (*sta)->a_prev = (*statement_tail);
      (*statement_tail) = *sta;
    } else {
      (*statement_head) = (*statement_tail) = *sta;
    }
  }
}

static void set_statement_block( g95x_statement *sta ) {
  int type = sta->type;

  if( ( type == ST_DERIVED_DECL ) 
   || ( type == ST_SUBROUTINE )
   || ( type == ST_FUNCTION )
   || ( type == ST_PROGRAM )
   || ( type == ST_BLOCK_DATA )
   || ( type == ST_MODULE )
   || ( type == ST_IF_BLOCK )
   || ( type == ST_DO )
   || ( type == ST_SELECT_CASE )
   || ( type == ST_WHERE_BLOCK )
   || ( type == ST_CRITICAL )
  )
    sta->block = g95_new_block;

  if( ( type == ST_ELSEIF )
   || ( type == ST_ELSE )
   || ( type == ST_CASE )
   || ( type == ST_ELSEWHERE )
   || ( type == ST_CONTAINS )
  ) {
    g95x_statement *sta1 = block_last(); /* CONTAINS without PROGRAM */
    if( sta1 )
      sta->block = sta1->block;
  }
    
  if( ( type == ST_END_TYPE ) 
   || ( type == ST_END_SUBROUTINE )
   || ( type == ST_END_FUNCTION )
   || ( type == ST_END_PROGRAM )
   || ( type == ST_END_BLOCK_DATA )
   || ( type == ST_END_MODULE )
   || ( type == ST_ENDIF )
   || ( type == ST_ENDDO )
   || ( type == ST_END_SELECT )
   || ( type == ST_END_WHERE )
   || ( type == ST_END_CRITICAL )
  ) {
    g95x_statement *sta1 = block_last(); /* END_PROGRAM without PROGRAM */
    if( sta1 )
      sta->block = sta1->block;
  }
}


static void fixup_ST_SUBROUTINE( g95x_statement* sta, g95_namespace* ns ) {
  g95x_statement *sta1 = block_last();
  if( sta1 && ( sta1->type == ST_INTERFACE ) ) {
    g95x_symbol_list *slist = g95x_get_symbol_list();
    slist->symbol = g95_state_stack->sym;
    if( sta1->u.interface.symbol_tail ) {
      sta1->u.interface.symbol_tail->next = slist;
      sta1->u.interface.symbol_tail = slist;
    } else {
      sta1->u.interface.symbol_head =
      sta1->u.interface.symbol_tail = slist;
    }
  }
  sta->u.subroutine.attr = *(g95x_get_current_attr());
}

static void fixup_ST_FUNCTION( g95x_statement* sta, g95_namespace* ns ) {
  fixup_ST_SUBROUTINE( sta, ns );
  sta->u.function.ts = *(g95x_get_current_ts());
}

static void fixup_ST_INTERFACE( g95x_statement* sta, g95_namespace* ns ) {
  sta->u.interface.type = current_interface.type;
}

static void fixup_ST_END_INTERFACE( g95x_statement* sta, g95_namespace* ns ) {
  g95x_statement *sta1 = block_last();
  sta->u.interface = sta1->u.interface;
  sta->u.interface.st_interface = sta1;
}

static void fixup_ST_FORMAT( g95x_statement* sta, g95_namespace* ns ) {
  sta->u.format.here = new_st.here;
}

static void fixup_ST_EQUIVALENCE( g95x_statement* sta, g95_namespace* ns ) {
  sta->u.equiv.e1 = ns->equiv;
  sta->u.equiv.e2 = ns->x.equiv1;
  ns->x.equiv1 = ns->equiv;
}

static void fixup_ST_DATA( g95x_statement* sta, g95_namespace* ns ) {
  sta->u.data.d1 = ns->data;
  sta->u.data.d2 = ns->x.data1;
  ns->x.data1 = ns->data;
}


static void fixup_ST_USE( g95x_statement* sta, g95_namespace* ns ) {
  ns->x.equiv1 = ns->equiv; 
  sta->u.use.only = g95x_get_only_flag();
}


static void fixup_ST_IMPLICIT( g95x_statement* sta, g95_namespace* ns ) {
  int i;
  g95_typespec *new_ts = g95x_get_implicit_ts();
  for( i = 0; i < G95_LETTERS; i++ ) 
    sta->u.implicit.ts[i] = new_ts[i];
}

  
static void fixup_ST_ENUM( g95x_statement* sta, g95_namespace* ns ) {
  sta->enum_bindc = g95x_enum_bindc();
}

static void fixup_ST_DATA_DECL( g95x_statement* sta, g95_namespace* ns ) {
  g95x_statement *sta1 = block_last();
  if( sta1 )
    sta->in_derived = sta1->type == ST_DERIVED_DECL;
  sta->u.data_decl.ts = *(g95x_get_current_ts());
  sta->u.data_decl.as = (struct g95_array_spec*)g95x_take_current_as();
  sta->u.data_decl.cas = (struct g95_coarray_spec*)g95x_take_current_cas();
  sta->u.data_decl.attr = *(g95x_get_current_attr());
  sta->u.data_decl.decl_list = g95x_take_decl_list();
}

static void fixup_ST_DERIVED_DECL( g95x_statement* sta, g95_namespace* ns ) {
  sta->u.data_decl.attr = *(g95x_get_current_attr());
}

static void fixup_ST_PARAMETER( g95x_statement* sta, g95_namespace* ns ) {
  sta->u.data_decl.decl_list = g95x_take_decl_list();
}

static void fixup_ST_ATTR_DECL( g95x_statement* sta, g95_namespace* ns ) {
  
  sta->u.data_decl.decl_list = g95x_take_decl_list();
  sta->u.data_decl.attr = *(g95x_get_current_attr());
  
}

static void fixup_ST_COMMON( g95x_statement* sta, g95_namespace* ns ) {
  sta->u.common.common_list = g95x_take_common_list();
}

static void fixup_ST_ENUMERATOR( g95x_statement* sta, g95_namespace* ns ) {
  sta->u.enumerator.enumerator_list = g95x_take_enumerator_list();
}

static void fixup_ST_CALL( g95x_statement* sta, g95_namespace* ns ) {
  int i;
  for( i = 0; i < sta->n_ext_locs; i++ ) /* the first locus may be a label */
    if( sta->ext_locs[i].type == G95X_LOCUS_NAME ) {
      sta->ext_locs[i].sub_code = sta->code;
      break;
    } 
  for( i = 0; i < sta->n_ext_locs; i++ ) 
    if( ( sta->ext_locs[i].type == G95X_LOCUS_KWRD )
     && ( sta->ext_locs[i].sub_code == NULL ) 
    ) 
      sta->ext_locs[i].sub_code = sta->code;
}

static void fixup_ST_ASSIGNMENT( g95x_statement* sta, g95_namespace* ns ) {
  int i;
  for( i = 0; i < sta->n_ext_locs; i++ ) 
    if( ( sta->ext_locs[i].type == G95X_LOCUS_INOP )
     && ( sta->ext_locs[i].u.inop.iop == INTRINSIC_ASSIGN )
    ) sta->ext_locs[i].ass_code = sta->code;
}
/*
 * Build and push a statement once it is matched. 
 */

g95x_statement *g95x_statement_head = NULL, 
               *g95x_statement_tail = NULL;

void g95x_push_statement( 
  g95_statement type, g95_code *code, g95x_locus *where 
) {
  g95x_statement *sta;
  g95_namespace *ns;


  switch( type ) {
    case ST_SIMPLE_IF:
    case ST_WHERE:
    case ST_FORALL: 
      where = &code->x.where;
    default:
    break;
  }

  ns = ( type == ST_MODULE_PROC ) ? g95_current_ns->parent : g95_current_ns;

    
  if( type == ST_IMPLIED_ENDDO ) {
    sta = ns->x.statement_tail;
    sta->implied_enddo++;
    goto block;
  }

  sta = get_statement_with_locs( where );

  sta->type = type;
  
  sta->ns = ns;

  sta->here = g95_statement_label;

  set_statement_code( sta, code );

  manage_statement_list( &ns->x.statement_head, &ns->x.statement_tail, &sta, 1 );
  manage_statement_list( &g95x_statement_head, &g95x_statement_tail, &sta, 0 );

  
  set_statement_block( sta );

  switch( type ) {
    case ST_SUBROUTINE:      fixup_ST_SUBROUTINE( sta, ns );      break;
    case ST_FUNCTION:        fixup_ST_FUNCTION( sta, ns );        break;
    case ST_INTERFACE:       fixup_ST_INTERFACE( sta, ns );       break;
    case ST_END_INTERFACE:   fixup_ST_END_INTERFACE( sta, ns );   break;
    case ST_FORMAT:          fixup_ST_FORMAT( sta, ns );          break;
    case ST_EQUIVALENCE:     fixup_ST_EQUIVALENCE( sta, ns );     break;
    case ST_DATA:            fixup_ST_DATA( sta, ns );            break;
    case ST_USE:             fixup_ST_USE( sta, ns );             break;
    case ST_IMPLICIT:        fixup_ST_IMPLICIT( sta, ns );        break;
    case ST_ENUM:            fixup_ST_ENUM( sta, ns );            break;
    case ST_DATA_DECL:       fixup_ST_DATA_DECL( sta, ns );       break;
    case ST_DERIVED_DECL:    fixup_ST_DERIVED_DECL( sta, ns );    break;
    case ST_PARAMETER:       fixup_ST_PARAMETER( sta, ns );       break;
    case ST_ATTR_DECL:       fixup_ST_ATTR_DECL( sta, ns );       break;
    case ST_COMMON:          fixup_ST_COMMON( sta, ns );          break;
    case ST_ENUMERATOR:      fixup_ST_ENUMERATOR( sta, ns );      break;
    case ST_CALL:            fixup_ST_CALL( sta, ns );            break;
    case ST_ASSIGNMENT:      fixup_ST_ASSIGNMENT( sta, ns );      break;
    default:
    break;
  }



block:
  block_manage( sta, type );


  switch( type ) {
    g95_code *c = NULL;
    case ST_FORALL: 
    case ST_SIMPLE_IF:
      c = sta->code->block;
      goto push;
    case ST_WHERE:
      c = sta->code->block->next;

push:      
      g95_statement_label = NULL;
      
      g95x_push_statement( c->x.sta, c, &c->x.where );
      
      g95_statement_label  = sta->here;
      
    break;
    default:
    break;
  }
  
  
  current_statement_locus_index = 0;
  
}

/*
 *
 */
 
int g95x_symbol_global( g95_symbol *sym, g95_namespace *ns ) {
  
  
  if( sym->attr.dummy )
    return 0;
  if( sym->x.imported )
    return 1;
  
  if( sym->module ) {
    if( ! strcmp( sym->module, "(intrinsic)" ) ) {
      return 1;
    } else if( ! strcmp( sym->module, "(global)" ) ) {
      return 1;
    } else if( strcmp( sym->module, ns->proc_name->name ) ) {
      return 1;
    }
  }

  if( sym->attr.access == ACCESS_PRIVATE ) {
    if( ( sym->attr.flavor == FL_PROCEDURE ) 
     && ( sym->attr.proc == PROC_MODULE )  ) {
      return 1;
    }
    return 0;
  }

  switch( sym->attr.flavor ) {
    case FL_PROCEDURE:
      switch( sym->attr.proc ) {
        case PROC_INTERNAL: case PROC_DUMMY:
	case PROC_ST_FUNCTION:
	  return 0;
	case PROC_MODULE: case PROC_INTRINSIC: case PROC_EXTERNAL:
	  return 1;
        default:
          return 1;
      }	
   case FL_PROGRAM: case FL_MODULE:
     return 1; 
   default:
     return 0;
  }
  
}

/*
 *  Take a substring of a statement; inspired from print_locs
 */


char* g95x_get_code_substring( 
  char *string, g95x_locus *xw, int start, int length, int name 
) {
  g95_linebuf *lb;
  int l = xw->lb1->linenum;
  int i = 0;
  int seen_name = 0;
  
  if( ( ! xw->lb1 ) 
   || ( xw->column1 == -1 ) 
   || ( xw->column2 == -1 ) 
  )
    g95x_die( "Attempt to take substring of an empty locus\n" );
  
  for( lb = xw->lb1; lb && ( lb->file == xw->lb1->file ) 
       && ( lb->linenum <= xw->lb2->linenum ); 
       lb = lb->next, l++ 
  ) {
    char *line = lb->line, *mask = lb->mask;
    int c1 = 1, c2 = lb->size + 1, c, d1 = -1, d2 = -1;
    
    if( lb == xw->lb1 ) 
      c1 = xw->column1;
    if( ( lb == xw->lb2 ) && ( xw->column2 < c2 ) ) 
      c2 = xw->column2;
    
    for( c = c1; c < c2; c++ ) {
      if( ( d1 < 0 ) && ( ( mask[c-1] == G95X_CHAR_CODE ) 
       || ( mask[c-1] == G95X_CHAR_STRG ) ) ) {
  	d1 = c;
      }
      if( ( d1 > 0 ) && ( ( mask[c-1] == G95X_CHAR_CODE ) 
       || ( mask[c-1] == G95X_CHAR_STRG ) ) ) {
  	d2 = c;
      }
      
      if( ( mask[c-1] == G95X_CHAR_CODE ) 
       || ( mask[c-1] == G95X_CHAR_STRG ) ) {
        if( name && ( line[c-1] == '(' ) && ( ! seen_name ) )
          continue;
        if( i >= start ) 
          string[i-start] = line[c-1];
        seen_name++;
        i++;
	if( i - start > length - 1 )
	  goto done;
      }
      
    }
     
    
  }

done:
  string[i] = '\0';
  
  if( name ) 
    for( i = 0; i < length; i++ ) {
      string[i] = string[i] >= 'A' && string[i] <= 'Z' ? string[i] - 'A' + 'a' : string[i];
      if( ( ! ( ( string[i] >= 'a' ) && ( string[i] <= 'z' ) ) )
       && ( ! ( ( string[i] >= '0' ) && ( string[i] <= '9' ) ) )
       && ( ! ( string[i] == '_' ) ) ) {
        string[i] = '\0';
        break;
      }
    } 
  
  
  return string;
}


/*
 * The aim of fixup symbols is to remove spurious symbols generated 
 * while parsing; eg :
 * PUBLIC SUBR
 * INTERFACE SUBR
 * END INTERFACE
 * generates a symbol whose name is SUBR, with no formal arguments
 * since we introduced alias-symbols, symbols accessible through alias are not
 * visible anymore
 */

//g95_generic_sym

static void _fixup_symbols( g95_symtree *st, void* data ) {
  g95_namespace *ns = data;
  g95_symbol *s = st->n.sym;

#ifdef OLD_FIXUP  
  
  g95_namespace *ns1;
  g95_symbol *s1 = NULL;
  g95_symtree *st1;
  g95_actual_arglist *ap = NULL;
  symbol_attribute *attr = &s->attr;


  if( 
      ( s->formal == NULL ) 
   && ( attr->allocatable == 0 )           && ( attr->dimension == 0 )             
   && ( attr->external == 0 )              && ( attr->intrinsic == 0 )             
   && ( attr->volatile_ == 0 )             && ( attr->optional == 0 )              
   && ( attr->pointer == 0 )               && ( attr->save == 0 )                  
   && ( attr->target == 0 )                && ( attr->dummy == 0 )                 
   && ( attr->result_var == 0 )            && ( attr->entry == 0 )                 
   && ( attr->value == 0 )                 && ( attr->async == 0 )                 
   && ( attr->procedure == 0 )             && ( attr->data == 0 )                  
   && ( attr->use_assoc == 0 )             && ( attr->equivalenced == 0 )          
   && ( attr->contained == 0 )             && ( attr->by_value == 0 )              
   && ( attr->desc == 0 )                  && ( attr->noinit == 0 )                
   && ( attr->only == 0 )                  && ( attr->protected == 0 )             
                                           && ( attr->untyped == 0 )               
   && ( attr->used == 0 )                  && ( attr->set == 0 )                   
   && ( attr->alloced == 0 )               && ( attr->dealloced == 0 )             
   && ( attr->st_construct == 0 )          && ( attr->hidden == 0 )                
   && ( attr->st_construct0 == 0 )         && ( attr->artificial == 0 )            
   && ( attr->current == 0 )               && ( attr->restore == 0 )               
   && ( attr->no_restore == 0 )            && ( attr->targetted == 0 )             
   && ( attr->aliasable == 0 )             && ( attr->st_dummy == 0 )              
   && ( attr->abstract == 0 )              && ( attr->invoked == 0 )               
   && ( attr->seen_formal == 0 )           && ( attr->used_formal == 0 )           
   && ( attr->in_namelist == 0 )           && ( attr->in_common == 0 )             
   && ( attr->entry_result == 0 )          && ( attr->dummy_proc == 0 )            
   && ( attr->sequence == 0 )              && ( attr->elemental == 0 )             
   && ( attr->pure == 0 )                  && ( attr->recursive == 0 )             
   && ( attr->bind == 0 )                  && ( attr->resolved == 1 )              
   && ( attr->modproc == 0 )               && ( attr->proc == PROC_UNKNOWN )       
   && ( attr->itype == ITYPE_NONE )        && ( attr->iproc == IPROC_NONE )        
   && ( attr->flavor == FL_PROCEDURE )     
   && ( attr->if_source == IFSRC_UNKNOWN ) && ( attr->intent == INTENT_UNKNOWN )   
   && ( ( ! s->module ) || strcmp( s->module, "(intrinsic)" ) ) 
  ) {
    for( ns1 = ns; ns1; ns1 = ns1->parent ) {
      st1 = g95_find_symtree( ns1->generic_root, st->name );
      if( st1 ) {
        s1 = g95_search_interface( st1->n.generic, s->ts.type == BT_UNKNOWN ? 1 : 0, &ap );
        if( s1 )
          break;
      }
      if( ns1->interface && ( ! ns1->import )  )
        break;
    }
    if( s1 )
      goto done;
    s->x.is_generic = 1;
  }
#else

  if( st->ambiguous )
   return;
   
  s->x.is_generic = g95_generic_sym( st->name, ns, s->ts.type == BT_UNKNOWN ? 1 : 0 );

#endif


  if( ( s->module && ( ! strcmp( s->module, "(intrinsic)" ) ) ) || s->attr.intrinsic )
    s->x.is_generic = 0;
  
  if( ! strcmp( st->name, s->name ) ) 
    s->x.visible++;
    
#ifdef OLD_FIXUP
done: 
#endif
  return; 
}


static void fixup_symbols( g95_namespace *ns ) {
  
  if( ns->sym_root == NULL )
    return;
  
  g95x_traverse_symtree( ns->sym_root, _fixup_symbols, ns );

}

static void _clear_generic( g95_symtree *st, void* data ) {
  g95_interface *intf = st->n.generic;
  for( ; intf; intf = intf->next ) 
    intf->sym->x.is_generic = 0;
}


static void clear_generics( g95_namespace *ns ) {
  
  if( ns->generic_root == NULL )
    return;
  
  g95x_traverse_symtree( ns->generic_root, _clear_generic, NULL );

}


void fixup_generics( g95_namespace *ns ) {
  g95_namespace *ns1;
  g95x_statement *sta;

  if( ! ns )
    return;

  for( ns1 = ns; ns1; ns1 = ns1->sibling ) {

    for( sta = ns1->x.statement_head; sta; sta = sta->next ) {
      if( sta->type == ST_INTERFACE ) {
        g95x_symbol_list *sl;
        for( sl = sta->u.interface.symbol_head; sl; sl = sl->next ) 
          fixup_generics( sl->symbol->formal_ns );
      }
      if( sta == ns1->x.statement_tail )
        break;
    }

    fixup_symbols( ns1 );
    clear_generics( ns1 );
    
    fixup_generics( ns1->contained );
  }
}

void g95x_fixup_generics( g95_namespace *ns ) {
  g95x_statement *sta;
  int i;
  char string[G95_MAX_SYMBOL_LEN+1];

  fixup_generics( ns );
  
  
  /* after all has been resolved */
  
  for( sta = g95x_statement_head; sta; sta = sta->a_next ) 
    for( i = 0; i < sta->n_ext_locs; i++ ) {
      g95x_ext_locus* xw = &sta->ext_locs[i];
      switch( xw->type ) {
        case G95X_LOCUS_NAME: {
        g95_expr *expr = xw->fct_expr;
        g95_code *code = xw->sub_code;

        
        if( expr || code ) {


          if( expr )
            expr = expr->x.full ? expr->x.full : expr;

          g95_symbol* symbol  = expr ? expr->symbol   : code->sym;
          g95_symbol* xsymbol = expr ? expr->x.symbol : code->x.symbol;

          
          
          if( symbol == NULL ) {
          
            xw->type = G95X_LOCUS_INTR;
            xw->u.intr.isym = g95x_get_intrinsic_by_symbol( xsymbol );
          
          
          } else {
          
            /* evil pointer cast */  
            g95_interface* intf = expr ? (g95_interface*)expr->x.generic : (g95_interface*)code->x.generic;
            if( intf ) {
              xw->type = G95X_LOCUS_GNRC;
              xw->u.gnrc.intf = intf;
              xw->u.gnrc.symbol = symbol;
            } else {

              xw->type = G95X_LOCUS_SMBL;
              xw->u.smbl.symbol = symbol;

            }
          
          }

        } else {
          void* p;
          
          
          g95x_get_code_substring( string, &xw->loc, 0, G95_MAX_SYMBOL_LEN, 1 );
          
          
          g95_symbol* symbol;
          if( sta->block && ! strcmp( sta->block->name, string ) ) { 
            symbol = sta->block;
          } else {
            g95x_resolve_sym( string, sta->ns, &p );
            symbol = (g95_symbol*)p; /* evil pointer cast */
          }
          
          
          
          if( ( symbol->module && ( strcmp( symbol->module, "(intrinsic)" ) == 0 ) ) 
           || ( symbol->attr.intrinsic ) 
          ) {
          
            xw->type = G95X_LOCUS_INTR;
            xw->u.intr.isym = g95x_get_intrinsic_by_symbol( symbol );
            
            
          } else {
          
            xw->type = G95X_LOCUS_SMBL;
            xw->u.smbl.symbol = symbol;
          
          }
          
        
        }
        
        break;
      } 
      
      case G95X_LOCUS_GNRC: {
        void *p;
        g95x_get_code_substring( string, &xw->loc, 0, G95_MAX_SYMBOL_LEN, 1 );
        g95x_resolve_gen( string, sta->ns, &p );
        
        if( p == NULL ) {
        
          if( ( sta->type == ST_ATTR_DECL ) 
           && ( sta->u.data_decl.attr.access != ACCESS_UNKNOWN ) 
          ) {
            /* Must be a symbol */
            
            xw->type = G95X_LOCUS_SMBL;
            g95x_resolve_sym( string, sta->ns, &p );
            
            xw->u.smbl.symbol = (g95_symbol*)p;  /* evil pointer cast */
            
          } else 
            g95x_die( "Unexpected null generic" );
        
          
        } else {
          xw->u.gnrc.intf = (g95_interface*)p;  /* evil pointer cast */
        }
        
        break;
      } 
      
      case G95X_LOCUS_USOP: {
        g95_expr *expr = xw->u.usop.expr;
        void *p;
        if( expr ) {
          expr = expr->x.full ? expr->x.full : expr;
          xw->u.usop.symbol = expr->symbol;
          xw->u.usop.uop = expr->x.uop;
        } else {
          g95x_get_code_substring( string, &xw->loc, 0, G95_MAX_SYMBOL_LEN, 1 );
          g95x_resolve_uop( string, sta->ns, &p );
          xw->u.usop.uop = (g95_symtree*)p; /* evil pointer cast */
        }
      
        break;
      } 
      
      
      case G95X_LOCUS_INOP: {

        if( xw->u.inop.iop == INTRINSIC_ASSIGN ) {
          g95_code *code = xw->ass_code;
          if( code && ( code->type == EXEC_CALL ) ) {
            xw->u.inop.intf = code->x.intr_op;
            xw->u.inop.symbol = code->sym;
          } else {
            void *p;
            g95x_resolve_iop( xw->u.inop.iop, sta->ns, &p );
            xw->u.inop.intf = (g95_interface*)p;
          }
        
        } else {
          g95_expr *expr = xw->u.inop.expr;
        
          if( expr && ( expr->type == EXPR_FUNCTION ) ) {
            expr = expr->x.full ? expr->x.full : expr;
            g95_symbol* s = expr->symbol ? expr->symbol : expr->x.symbol;
            xw->u.inop.intf = expr->x.iop;
            xw->u.inop.symbol = s;
          } else {
            void *p;
            g95x_resolve_iop( xw->u.inop.iop, sta->ns, &p );
            xw->u.inop.intf = (g95_interface*)p;
          } 
          
        } 
        
        break;
      }
      default:
      break;
    }
  }  

}










