#include "g95.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>


#define intrinsic_alias_id( isym ) (char*)(isym) + 1

#ifdef G95X_64BITS
const char* REFFMT0 = "0x%llx";
const char* REFFMT1 = "%16.16llx";
#else
const char* REFFMT0 = "0x%x";
const char* REFFMT1 = "%8.8x";
#endif


static void print_statement( g95x_statement* );
static void print_expr( g95_expr*, int );
static void print_ref( g95_ref* );
static void print_constructor( g95_constructor* );
static void print_code( g95x_statement* );
static void print_symbol( g95_symbol*, g95_namespace* );
static void print_component( g95_component* );   
static void print_io_var( g95_code* );
static void print_data_var( g95_data_variable* );
static void print_symbol_name( g95_symbol*, g95_namespace* );
static void print_labels( g95_namespace* );

static const char* sort_flavor( symbol_attribute *, int );
static const char *sort_ref( g95_ref * );
static const char *sort_statement( g95x_statement * );
static const char *sort_interface( interface_type );
static const char *sort_iop( g95_intrinsic_op );
static const char *sort_expr( g95_expr* );
static void print_constant( g95_expr*, int );
static void* type_id( g95_typespec *ts );
static int valid_namespace( g95_namespace *ns );
static void print_symbol_alias( g95x_locus* xw, void* generic, g95_symbol* symbol, const char* name );


#define symbol_intrinsic( symbol ) \
  ( ( (symbol)->module && ( ! strcmp( (symbol)->module, "(intrinsic)" ) ) ) \
  || (symbol)->attr.intrinsic )
  

g95_intrinsic_sym *symbol_intrinsic_id( char* name ) {
  g95_intrinsic_sym *isymf = g95_find_function( name );
  g95_intrinsic_sym *isyms = g95_find_subroutine( name );
  return isymf ? isymf : isyms;
}

g95_intrinsic_sym* g95x_get_intrinsic_by_symbol( g95_symbol *sym ) {
  g95_intrinsic_sym *isym = NULL, *isymf, *isyms;
  
  if( ! symbol_intrinsic( sym ) )
    g95x_die( "Expected intrinsic" );
  
  
  isymf = g95_find_function( sym->name );
  isyms = g95_find_subroutine( sym->name );
  
  if( sym->attr.function ) 
    isym = isymf;
  else if( sym->attr.subroutine )
    isym = isyms;
    
  if( isym == NULL )
    isym = isymf ? isymf : isyms;
  
  if( isym == NULL )
    g95x_die( "Intrinsic not found" );
  
  return isym;
}


static int _print_expr( g95_expr *e, void *data ) {


  g95x_bbt_id* st;
  if( ( st = g95x_get_bbt_id( (g95x_pointer)e, NULL, &root_bbt_id ) ) )
    if( st->defined )
      return -1;

  
  if( g95x_option.reduced_exprs ) {
    print_expr( e, 1 );
    return -1;
  } else {
    print_expr( e, 0 );
    return +1;
  }
}

static int _print_ref( g95_ref *r, void *data ) {
  print_ref( r );
  return 1;
}

static int _print_constructor( g95_constructor *c, void *data ) {
  print_constructor( c );
  return 1;
}


typedef struct _print_symbol_data {
  g95_namespace *ns;
} _print_symbol_data;

int g95x_symbol_valid( g95_symbol* sym ) {
    
  if( sym->attr.flavor == FL_UNKNOWN )
    return 0;
    
  if( sym->x.is_generic ) 
    return 0;   


  return 1;
}



static int _print_symbol( g95_symbol* sym, void* data ) {
  _print_symbol_data* psd = data;
  g95_namespace* ns = psd->ns;
  g95_component* k;
  g95x_bbt_id* st;

  
  if( ! g95x_symbol_valid( sym ) )
    return -1;

  if( symbol_intrinsic( sym ) ) {
    g95_intrinsic_sym *isym = g95x_get_intrinsic_by_symbol( sym ); 
                                                  
    if( ( st = g95x_get_bbt_id( (g95x_pointer)isym, NULL, &root_bbt_id ) ) )
      if( st->defined )
        return -1;
    
  }


  if( ( st = g95x_get_bbt_id( (g95x_pointer)sym, NULL, &root_bbt_id ) ) )
    if( st->defined )
      return -1;


  if( sym->x.imported && ( ! symbol_intrinsic( sym ) ) ) {
    g95x_print( 
      "<symbol id=\"%R\" flavor=\"%s\" name=\"", 
      sym, sort_flavor( &sym->attr, 1 ) 
    );
    
    print_symbol_name( sym, ns );
    g95x_print( "\" imported=\"1\"" );

    if( sym->x.visible == 0 ) 
      g95x_print( " hidden=\"1\"" );
    
    if( sym->attr.flavor == FL_DERIVED ) {
      g95x_print( " components=\"" );
      for( k = sym->components; k; k = k->next ) {
        g95x_print( "%R", k );
        if( k->next )
          g95x_print( "," );
      }
      g95x_print( "\"" );
    }
    
    
    g95x_print( "/>\n" );
    
    
    if( sym->attr.flavor == FL_DERIVED ) { /* we need components definition for imported 
                                              derived types */
      if( sym->components )    
        for( k = sym->components; k; k = k->next ) {
          g95x_print( "<component id=\"%R\" name=\"%s\"/>\n", k, k->name );
        }
    }
    
  } else
    print_symbol( sym, ns ); 


  if( sym->attr.flavor == FL_DERIVED ) {
    g95_typespec ts;
    ts.type = BT_DERIVED;
    ts.derived = sym;
    g95x_print( 
      "<type id=\"%R\" type=\"DERIVED\" symbol=\"%R\"/>\n", 
      type_id( &ts ), sym
    );
  }

  
  return 1; 
}

static int _print_component( g95_component* c, void* data ) {
  print_component( c ); 
  return 1; 
}

static int _print_io_var( g95_code *c, void* data ) {
  print_io_var( c );
  return +1;
}

static int _print_data_var( g95_data_variable *var, void* data ) {
  print_data_var( var );
  return 1;
}




/*
 * Returns a const char* per statement type
 */

static const char *sort_statement( g95x_statement *sta ) {
const char *p = NULL;

  switch( sta->type ) {
  case ST_ARITHMETIC_IF:  p = "ARITHMETIC_IF";         break;
  case ST_ALLOCATE:       p = "ALLOCATE";              break;
  case ST_ATTR_DECL: {
     symbol_attribute *a = &sta->u.data_decl.attr;
     if( a->access == ACCESS_PUBLIC )   p = "PUBLIC";
     if( a->access == ACCESS_PRIVATE )  p = "PRIVATE";
     if( a->allocatable )               p = "ALLOCATABLE";
     if( a->dimension )                 p = "DIMENSION";
     if( a->external )                  p = "EXTERNAL";
     if( a->intent == INTENT_IN )       p = "INTENT_IN";
     if( a->intent == INTENT_OUT )      p = "INTENT_OUT";
     if( a->intent == INTENT_INOUT )    p = "INTENT_INOUT";
     if( a->intrinsic )                 p = "INTRINSIC";
     if( a->optional )                  p = "OPTIONAL";
     if( a->pointer )                   p = "POINTER";
     if( a->save )                      p = "SAVE";
     if( a->target )                    p = "TARGET";
     if( a->protected )                 p = "PROTECTED";
     if( a->bind )                      p = "BIND";
     if( a->value )                     p = "VALUE";
     if( a->volatile_ )                 p = "VOLATILE";
     if( a->async )                     p = "ASYNCHRONOUS";
     
     if( p == NULL )
       goto err;
     

    break;
err:
    g95x_die( "Could not find out statement type\n" );
  }
  case ST_BACKSPACE:      p = "BACKSPACE";             break;
  case ST_BLOCK_DATA:     p = "BLOCK_DATA";            break;
  case ST_CALL:           p = "CALL";                  break;
  case ST_CASE:           p = "CASE";                  break;
  case ST_CLOSE:          p = "CLOSE";                 break;
  case ST_COMMON:         p = "COMMON";                break;
  case ST_CONTINUE:       p = "CONTINUE";              break;
  case ST_CONTAINS:       p = "CONTAINS";              break;
  case ST_CYCLE:          p = "CYCLE";                 break;
  case ST_DATA_DECL: {
    if( sta->in_derived )
      p = "COMPONENT_DEF";
    else
      p = "TYPE_DECLARATION";      
    break;
  }
  case ST_DATA:           p = "DATA";                  break;
  case ST_DEALLOCATE:     p = "DEALLOCATE";            break;
  case ST_DERIVED_DECL:   p = "DERIVED_TYPE_DEF";      break;
  case ST_DO:             p = "DO";                    break;
  case ST_ELSE:           p = "ELSE";                  break;
  case ST_ELSEIF:         p = "ELSE_IF";               break;
  case ST_ELSEWHERE:      p = "ELSE_WHERE";            break;
  case ST_END_BLOCK_DATA: p = "END_BLOCK_DATA";        break;
  case ST_ENDDO:          p = "END_DO";                break;
  case ST_END_ENUM:       p = "END_ENUM";              break;
  case ST_END_FILE:       p = "ENDFILE";               break;
  case ST_END_FORALL:     p = "END_FORALL";            break;
  case ST_END_FUNCTION:   p = "END_FUNCTION";          break;
  case ST_ENDIF:          p = "END_IF";                break;
  case ST_END_INTERFACE:  p = "END_INTERFACE";         break;
  case ST_END_MODULE:     p = "END_MODULE";            break;
  case ST_END_PROGRAM:    p = "END_PROGRAM";           break;
  case ST_END_SELECT:     p = "END_SELECT_CASE";       break;
  case ST_END_SUBROUTINE: p = "END_SUBROUTINE";        break;
  case ST_END_WHERE:      p = "END_WHERE";             break;
  case ST_END_TYPE:       p = "END_DERIVED_TYPE_DEF";  break;
  case ST_ENTRY:          p = "ENTRY";                 break;
  case ST_ENUM:           p = "ENUM";                  break;
  case ST_ENUMERATOR:     p = "ENUMERATOR";            break;
  case ST_EQUIVALENCE:    p = "EQUIVALENCE";           break;
  case ST_EXIT:           p = "EXIT";                  break;
  case ST_FLUSH:          p = "FLUSH";                 break;
  case ST_FORALL_BLOCK:   p = "FORALL_CONSTRUCT";      break;
  case ST_FORALL:         p = "FORALL";                break;
  case ST_FORMAT:         p = "FORMAT";                break;
  case ST_FUNCTION:       p = "FUNCTION";              break;
  case ST_GOTO:           p = "GOTO";                  break;
  case ST_IF_BLOCK:       p = "IF_THEN";               break;
  case ST_IMPLICIT:       p = "IMPLICIT";              break;
  case ST_IMPLICIT_NONE:  p = "IMPLICIT_NONE";         break;
  case ST_IMPLIED_ENDDO:  
    g95x_die( "Unexpected ST_IMPLIED_ENDDO" );
  break;
  case ST_IMPORT:         p = "IMPORT";                break;
  case ST_INQUIRE:        p = "INQUIRE";               break;
  case ST_INTERFACE:      p = "INTERFACE";             break;
  case ST_PARAMETER:      p = "PARAMETER";             break;
  case ST_PRIVATE:        p = "PRIVATE";               break;
  case ST_PUBLIC:         p = "PUBLIC";                break;
  case ST_MODULE:         p = "MODULE";                break;
  case ST_MODULE_PROC:    p = "MODULE_PROCEDURE";      break;
  case ST_NAMELIST:       p = "NAMELIST";              break;
  case ST_NULLIFY:        p = "NULLIFY";               break;
  case ST_OPEN:           p = "OPEN";                  break;
  case ST_PAUSE:          p = "PAUSE";                 break;
  case ST_PROGRAM:        p = "PROGRAM";               break;
  case ST_READ:           p = "READ";                  break;
  case ST_RETURN:         p = "RETURN";                break;
  case ST_REWIND:         p = "REWIND";                break;
  case ST_STOP:           p = "STOP";                  break;
  case ST_SUBROUTINE:     p = "SUBROUTINE";            break;
  case ST_TYPE:           
    g95x_die( "Unexpected ST_TYPE" );               
  break;
  case ST_USE:            p = "USE";                   break;
  case ST_WAIT:           p = "WAIT";                  break;
  case ST_WHERE_BLOCK:    p = "WHERE_CONSTRUCT";       break;
  case ST_WHERE:          p = "WHERE";                 break;
  case ST_WRITE: 
    if( sta->u.is_print ) 
      p = "PRINT";  
    else
      p = "WRITE";               
  break;
  case ST_ASSIGN:               p = "ASSIGN";               break;
  case ST_ASSIGNMENT:           p = "ASSIGNMENT";           break;
  case ST_POINTER_ASSIGNMENT:   p = "POINTER_ASSIGNMENT";   break;
  case ST_SELECT_CASE:          p = "SELECT_CASE";          break;
  case ST_SEQUENCE:             p = "SEQUENCE";             break;
  case ST_SIMPLE_IF:            p = "IF";                   break;
  case ST_STATEMENT_FUNCTION:   p = "STMT_FUNCTION";        break;
  
  case ST_CRITICAL:             p = "CRITICAL";             break;
  case ST_END_CRITICAL:         p = "END_CRITICAL";         break;
  case ST_SYNC_ALL:             p = "SYNC_ALL";             break;
  case ST_SYNC_IMAGES:          p = "SYNC_IMAGES";          break;
  case ST_SYNC_TEAM:            p = "SYNC_TEAM";            break;
  case ST_SYNC_MEMORY:          p = "SYNC_MEMORY";          break;
  case ST_ALL_STOP:             p = "ALL_STOP";             break;
  case ST_NONE:
    g95x_die( "Bad statement code for statement %R", sta );
  }

  return p;
}

static const char *sort_interface( interface_type it ) {
  const char *p = NULL;

  switch( it ) {
    case INTERFACE_NAMELESS:      p = "NAMELESS";      break;
    case INTERFACE_GENERIC:       p = "GENERIC";       break;
    case INTERFACE_INTRINSIC_OP:  p = "INTRINSIC_OP";  break;
    case INTERFACE_USER_OP:       p = "USER_OP";       break;
    case INTERFACE_ABSTRACT:      p = "ABSTRACT";      break;
  }

  return p;
}

/* Intrinsic ops */

static const char *sort_iop( g95_intrinsic_op e ) {
const char *p = "";

  switch( e ) {
    case INTRINSIC_UPLUS:  p = "UPLUS";      break;    
    case INTRINSIC_UMINUS: p = "UMINUS";     break;
    case INTRINSIC_PLUS:   p = "PLUS";       break;
    case INTRINSIC_MINUS:  p = "MINUS";      break;           
    case INTRINSIC_TIMES:  p = "TIMES";      break;
    case INTRINSIC_DIVIDE: p = "DIVIDE";     break;        
    case INTRINSIC_POWER:  p = "POWER";      break;    
    case INTRINSIC_CONCAT: p = "CONCAT";     break; 
    case INTRINSIC_AND:    p = "AND";        break;           
    case INTRINSIC_OR:     p = "OR";         break;       
    case INTRINSIC_EQV:    p = "EQV";        break;    
    case INTRINSIC_NEQV:   p = "NEQV";       break;  
    case INTRINSIC_EQ:     p = "EQ";         break;     
    case INTRINSIC_NE:     p = "NE";         break;
    case INTRINSIC_GT:     p = "GT";         break;   
    case INTRINSIC_GE:     p = "GE";         break;
    case INTRINSIC_LT:     p = "LT";         break;     
    case INTRINSIC_LE:     p = "LE";         break;          
    case INTRINSIC_NOT:    p = "NOT";        break;
    case INTRINSIC_USER:   p = "USER";       break;
    case INTRINSIC_ASSIGN: p = "ASSIGNMENT"; break;
    case INTRINSIC_PAREN:  p = "PAREN";      break;
    default:
      g95x_die( "Bad op code" );
  }

  return p;
}

/* user ops */

typedef struct _symtree_data {
  struct _symtree_data *next;
  char* name;
} _symtree_data;


_symtree_data *symtree_data_uop_head = NULL;
_symtree_data *symtree_data_gen_head = NULL;

static _symtree_data *sort_symtree( const char* name, _symtree_data *head ) {
  _symtree_data *sud1;
  
  for( sud1 = head; sud1; sud1 = sud1->next ) {
    if( ! strcmp( name, sud1->name ) )
      return sud1; 
  }
    
  return NULL;
}

static _symtree_data* def_sort_symtree( const char* name, _symtree_data **head ) {
  _symtree_data *sud1;
  int len;
  
  sud1 = g95_getmem( sizeof( _symtree_data ) );
  len = strlen( name );
  sud1->name = g95_getmem( len + 1 );
  memcpy( sud1->name, name, len + 1 );
  
  sud1->next = *head;
  *head = sud1;
  
  return sud1;
}


static void _def_symtree0( g95_symtree *st, void* data ) {
  _symtree_data **head = (_symtree_data**)data;
  
  if( st == NULL )
    return;
  
  if( sort_symtree( st->name, *head ) == NULL ) 
    def_sort_symtree( st->name, head );
  
}

static void def_symtree( g95_namespace *ns, int stt, _symtree_data **head ) {
  g95_symtree *st;
  if( ! valid_namespace( ns ) )
    return;
  for( ; ns; ns = ns->sibling ) {
    if( ns->contained )
      def_symtree( ns->contained, stt, head );
    switch( stt ) {
      case 0:
        st = ns->uop_root;
      break;
      case 1:
        st = ns->generic_root;
      break;
      default:
        g95x_die( "Unexpected stt\n" );
      break;
    }
    if( st )
      g95x_traverse_symtree( st, _def_symtree0, head );


    g95x_statement *sta;
    for( sta = ns->x.statement_head; ; sta = sta->next ) {
      if( sta->type == ST_INTERFACE ) {
        g95x_symbol_list *sl;
        for( sl = sta->u.interface.symbol_head; sl; sl = sl->next ) 
          def_symtree( sl->symbol->formal_ns, stt, head );
      }
      if( sta == ns->x.statement_tail )
        break;
    }
  }
}

/* ref */

static const char *sort_ref( g95_ref *r ) {
  const char *p = "";
  
  switch( r->type ) {
    case REF_ARRAY: {
      switch( r->u.ar.type ) {
        case AR_FULL:
	  p = "ARRAY_FULL";
	break;
	case AR_ELEMENT:
	  p = "ARRAY_ELEMENT";
	break;
	case AR_SECTION:
	  p = "ARRAY_SECTION";
	break;
	case AR_UNKNOWN:
	  g95x_die( "Unknown array ref %R", r );
	break;
      }
    } break;
    case REF_COMPONENT: 
      p = "COMPONENT";
    break;
    case REF_SUBSTRING:
      p = "SUBSTRING";
    break;
    case REF_COARRAY:
      p = "COARRAY";
    break;
  }
  return p;
}


static const char* sort_flavor( symbol_attribute *a, int check ) {
const char *p = NULL;

  switch( a->flavor ) {
    case FL_PROGRAM:
      p = "PROGRAM";
    break;
    case FL_BLOCK_DATA:
      p = "BLOCK_DATA";
    break;
    case FL_MODULE:
      p = "MODULE";
    break;
    case FL_VARIABLE:
      p = "VARIABLE";
    break;
    case FL_PARAMETER:
      p = "PARAMETER";
    break;
    case FL_LABEL:
      p = "LABEL";
    break;
    case FL_PROCEDURE:
      if( a->function ) { 
        p = "FUNCTION";
      } else if( a->subroutine ) {
        p = "SUBROUTINE";
      } else {
        p = "PROCEDURE";
      }
    break;
    case FL_DERIVED:
      p = "DERIVED";
    break;
    case FL_NAMELIST:
      p ="NAMELIST";
    break;
    case FL_UNKNOWN:
      if( check )
        g95x_die( "Unknown flavor" );
    break;
  }

  return p;
}

static const char *sort_expr( g95_expr* e ) {
const char *p = NULL;

  switch( e->type ) {
    case EXPR_OP:        p = "OP";             break;
    case EXPR_FUNCTION:  {
      if( e->x.is_operator ) {
        p = "OP";
      } else {
        p = "FUNCTION"; 
      }      
      break;
    }
    case EXPR_CONSTANT:  p = "CONSTANT";       break;
    case EXPR_VARIABLE:  p = "VARIABLE";       break;
    case EXPR_SUBSTRING: p = "SUBSTRING";      break;
    case EXPR_STRUCTURE: p = "STRUCTURE";      break;
    case EXPR_PROCEDURE: p = "PROCEDURE";      break;
    case EXPR_ARRAY:     p = "ARRAY";          break;
    case EXPR_NULL:      p = "NULL";           break;
    default:
      g95x_die( "Bad expr code for %R", e );
  }

  return p;
}

/*
 *
 * Here, we prints locs in a format perl can easily eval
 * If ns is not NULL, then we resolve the symbol/uop/generic/common,
 * and print its id.
 * We use the mask to split location among different lines.
 *
 */



void g95x_resolve_iop( 
 g95_intrinsic_op type, g95_namespace *ns, void** p
) {
  *p = NULL;
  while( ns ) {
    *p = ns->operator[type];
    if( *p )
      return;
    if( ns->interface && ( ! ns->import )  )
      break;
    ns = ns->parent;
  }
  *p = (void*)sort_iop( type );
}

void g95x_resolve_uop( 
 char* name, g95_namespace *ns, void** p
) {
  *p = NULL;
  while( ns ) {
    *p = g95_find_symtree( ns->uop_root, name );
    if( *p )
      return;
    if( ns->interface && ( ! ns->import )  )
      break;
    ns = ns->parent;
  }
  _symtree_data *sud = sort_symtree( name, symtree_data_uop_head );
  if( sud )
    *p = (void*)sud->name;
}

void g95x_resolve_gen( 
 char* name, g95_namespace *ns, void** p
) {
  *p = NULL;
  while( ns ) {
    g95_symtree *st = g95_find_symtree( ns->generic_root, name );
    *p = st;
    if( *p )
      return;
    if( ns->interface && ( ! ns->import )  )
      break;
    ns = ns->parent;
  }
  _symtree_data *sud = sort_symtree( name, symtree_data_gen_head );
  if( sud )
    *p = (void*)sud->name;
}

void g95x_resolve_sym( 
 char* name, g95_namespace *ns, void** p
) {
  g95_symbol *s;
  *p = NULL;
  while( ns ) {
    if( g95_find_symbol( name, ns, 0, &s ) == 0 ) {
      if( s && g95x_symbol_valid( s ) )
        *p = s;
    }
    if( *p )
      return;
    if( ns->interface && ( ! ns->import )  )
      break;
    ns = ns->parent;
  }
}

/*
 * Print locs and sets the (l1,c1)-(l2-c2) of xr
 *
 */

#define NXR_MAX 10000

static void find_loc(
  g95x_locus *xw,
  g95x_locus *xr, int *nxr
) {
  g95_linebuf *lb;
  int l = xw->lb1->linenum;
  int ixr;
  
  *nxr = 0;
  
  for( ixr = 0; ixr < NXR_MAX; ixr++ ) {
      xr[ixr].lb1 = NULL; xr[ixr].column1 = -1; 
      xr[ixr].lb2 = NULL; xr[ixr].column2 = -1;
  }
  ixr = 0;
  
  for( lb = xw->lb1; lb && ( lb->file == xw->lb1->file ) 
       && ( lb->linenum <= xw->lb2->linenum ); 
       lb = lb->next, l++ 
  ) {
    char *mask = lb->mask;
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
      
    }
     
    if( ( d1 > 0 ) && xr ) {
      xr[ixr].lb1 = lb; xr[ixr].column1 = d1; 
      xr[ixr].lb2 = lb; xr[ixr].column2 = d2+1;
      ixr++;
      *nxr = ixr;
    }
    
  }

}

static void print_locs( g95x_locus *xw ) {
  g95x_locus xrt[NXR_MAX];
  int nxr;
  int ixr;
  
  if( ! xw->lb1 )
    g95x_die( "Attempt to print a null loc\n" );
    
  find_loc( xw, xrt, &nxr );
  
  g95x_print( "[" );
  
  for( ixr = 0; ixr < nxr; ixr++ ) {
    g95x_print( "%d,%d,%d,%d", 
      xrt[ixr].lb1->linenum-1, xrt[ixr].column1-1, 
      xrt[ixr].lb1->linenum-1, xrt[ixr].column2-1 
    );
    if( ixr < ( nxr - 1 ) ) 
      g95x_print( "," );
  }

  g95x_print( "]" );

}





static int find_ext_loc(
  g95x_ext_locus *xw,
  char* name, g95x_locus *xr, int *nxr
) {
  g95_linebuf *lb;
  int l = xw->loc.lb1->linenum;
  int i = 0;
  int ixr;
  
  *nxr = 0;
  
  for( ixr = 0; ixr < NXR_MAX; ixr++ ) {
      xr[ixr].lb1 = NULL; xr[ixr].column1 = -1; 
      xr[ixr].lb2 = NULL; xr[ixr].column2 = -1;
  }
  ixr = 0;
  
  for( lb = xw->loc.lb1; lb && ( lb->file == xw->loc.lb1->file ) 
       && ( lb->linenum <= xw->loc.lb2->linenum ); 
       lb = lb->next, l++ 
  ) {
    char *line = lb->line, *mask = lb->mask;
    int c1 = 1, c2 = lb->size + 1, c, d1 = -1, d2 = -1;
    
    
    if( lb == xw->loc.lb1 ) 
      c1 = xw->loc.column1;
    if( ( lb == xw->loc.lb2 ) && ( xw->loc.column2 < c2 ) ) 
      c2 = xw->loc.column2;
    
    for( c = c1; c < c2; c++ ) {
      if( ( d1 < 0 ) && ( ( mask[c-1] == G95X_CHAR_CODE ) 
       || ( mask[c-1] == G95X_CHAR_STRG ) ) ) {
        d1 = c;
      }
      if( ( d1 > 0 ) && ( ( mask[c-1] == G95X_CHAR_CODE ) 
       || ( mask[c-1] == G95X_CHAR_STRG ) ) ) {
        d2 = c;
      }
      
      if( name ) {
      
        if( mask[c-1] == G95X_CHAR_CODE ) {
          name[i] = line[c-1];
          if( ( name[i] >= 'A' ) && ( name[i] <= 'Z' ) ) {
            name[i] = 'a' + ( name[i] - 'A' );
          }
          i++;
        }
      
      }
    }
     
    if( ( d1 > 0 ) && xr ) {
      xr[ixr].lb1 = lb; xr[ixr].column1 = d1; 
      xr[ixr].lb2 = lb; xr[ixr].column2 = d2+1;
      ixr++;
      *nxr = ixr;
    }
    
  }
  
  if( name )
    name[i] = '\0';

  
  return -1;
}



static void print_ext_loc_name( 
  g95x_ext_locus *xw, g95_namespace *ns
) {
  g95x_locus xrt[NXR_MAX];
  int nxr;
  char name[G95_MAX_SYMBOL_LEN+1];


  find_ext_loc(
    xw, name, xrt, &nxr
  );

  switch( xw->type ) {
    case G95X_LOCUS_SMBL:
      print_symbol_alias( &xw->loc, NULL, xw->u.smbl.symbol, NULL );
    break;
    default:
      g95x_die( "Unexpected locus type" );
  }


}


static void print_ext_locs0(
  g95x_locus *xrt, int nxr
) {
  int ixr;
  for( ixr = 0; ixr < nxr; ixr++ ) {
    g95x_print( "%d,%d,%d,%d", 
      xrt[ixr].lb1->linenum-1, xrt[ixr].column1-1, 
      xrt[ixr].lb1->linenum-1, xrt[ixr].column2-1 
    );
    if( ixr < ( nxr - 1 ) ) 
      g95x_print( "," );
  }
}


static int print_ext_locs( 
  g95x_ext_locus *xw, g95_namespace *ns, g95x_locus *xr
) {
  g95x_locus xrt[NXR_MAX];
  int nxr;
  char name[G95_MAX_SYMBOL_LEN+1];
  
  if( ( ! xw->loc.lb1 ) || ( ! xw->loc.lb2 ) )
    g95x_die( "Attempt to print a null loc\n" );
    
  
  
  switch( xw->type ) {
    case G95X_LOCUS_BOZZ:
    case G95X_LOCUS_STRG:
    case G95X_LOCUS_HLLR:
      find_ext_loc(
        xw, NULL, xrt, &nxr
      );
    break;
    default:
      find_ext_loc(
        xw, name, xrt, &nxr
      );
  }
  
  g95x_print( "[" );
  print_ext_locs0( xrt, nxr );



  switch( xw->type ) {
    case G95X_LOCUS_GNRC:
      g95x_print( ",'%R'", xw->u.gnrc.intf );
      if( xw->u.gnrc.symbol )
        g95x_print( ",'%R'", xw->u.gnrc.symbol );
    break;
    case G95X_LOCUS_SMBL: 
      g95x_print( ",'" );
      print_symbol_alias( &xw->loc, NULL, xw->u.smbl.symbol, NULL );
      g95x_print( "'" );
    break;
    case G95X_LOCUS_INTR: 
      g95x_print( ",'%R'", xw->u.intr.isym );
    break;
    case G95X_LOCUS_CMMN: 
      g95x_print( ",'%R'", xw->u.cmmn.common ); 
    break;
    case G95X_LOCUS_KWRD: 
      g95x_print( ",'K'" );	
    break;
    case G95X_LOCUS_MMBR:
      g95x_print( ",'%R'", xw->u.mmbr.component );
    break;
    case G95X_LOCUS_ASSC: 
      if( xw->u.assc.symbol )
        g95x_print( ",'%R','A'", xw->u.assc.symbol );
      else if( xw->u.assc.intf )
        g95x_print( ",'%R','A'", xw->u.assc.intf );
      else
        g95x_die( "" );
    break;
    case G95X_LOCUS_BOZZ:
      g95x_print( ",'B'" );
    break;
    case G95X_LOCUS_STRG:
      g95x_print( ",'S'" );
    break;
    case G95X_LOCUS_HLLR:
      g95x_print( ",'H'" );
    break;
    case G95X_LOCUS_LABL:
      g95x_print( ",'%R'", xw->u.labl.label );
    break;
    case G95X_LOCUS_USOP:
      g95x_print( ",'%R'", xw->u.usop.uop );
      if( xw->u.usop.symbol ) 
        g95x_print( ",'%R'", xw->u.usop.symbol );
    break;
    case G95X_LOCUS_INOP: 
      g95x_print( ",'%R'", xw->u.inop.intf );
      if( xw->u.inop.symbol ) 
        g95x_print( ",'%R'", xw->u.inop.symbol );
    break;
    default:
      g95x_die( "Unexpected locus\n" );
    break;
  }
  
  g95x_print( "]" );

  return 1;
}



/* 
 * Prints xml stuff and takes care of indentation
 * Here, we handle only %s, %d, %x, %c from the standard printf
 * we add :
 * - %R : reference
 * - %T : reference without 0x
 */


int g95x_print( const char *fmt, ... ) {
  va_list ap;
  int n = 0;
  int i;
  static int indent = 0;
  int d = 0;
  
  if( g95x_option.indent ) {
    for( i = 0; fmt[i]; i++ ) {
      if( fmt[i] == '<' ) d++;
    }
    for( i = 0; fmt[i]; i++ ) {
      if( ( fmt[i] == '<' ) && ( fmt[i+1] == '/' ) ) indent--;
    }
    if( d ) {
      for( i = 0; i < indent; i++ ) {
        fprintf( g95x_option.g95x_out, "  " );
        n += 2;
      }
    }
  }
  
  /* %c, %d, %x, %s */
  {
    char* s; int n; char c; 
    g95x_pointer p, q;
    g95x_bbt_id *st = NULL;

    int i, j;
    int l;
    FILE* out = g95x_option.g95x_out;
    static char FMT256[256]; /* Handle most case with static buf */
    char* FMT;
    
    l = strlen( fmt );
    if( l < 256 ) 
      FMT = FMT256;
    else
      FMT = g95_getmem( l + 1 );

    strcpy( FMT, fmt );
    
    va_start( ap, fmt );
    
    for( i = 0; ; ) {
      for( j = i; j < l; j++ ) {
        if( ( FMT[j] == '%' ) && ( j < l - 1 ) ) {
	  FMT[j] = '\0';
	  fprintf( out, FMT + i );
	
          switch( FMT[j+1] ) {
	    case 'd':
	      n = va_arg( ap, int );
	      fprintf( out, "%d", n );
	    break;
	    case 'x':
	      n = va_arg( ap, int );
	      fprintf( out, "%x", n );
	    break;
	    case 'c':
	      c = va_arg( ap, int );
	      fprintf( out, "%c", c );
	    break;
	    case 's':
	      s = va_arg( ap, char* );
	      fprintf( out, "%s", s );
	    break;
	    case 'R':
	      p = va_arg( ap, g95x_pointer );
              st = g95x_get_bbt_id( p, NULL, &root_bbt_id );
              if( st == NULL )
                st = g95x_new_bbt_id( p, NULL, &root_bbt_id );
              if( ( FMT[j-5] == ' ' ) && ( FMT[j-4] == 'i' ) 
               && ( FMT[j-3] == 'd' ) && ( FMT[j-2] == '=' ) 
               && ( FMT[j-1] == '"' )
              ) st->defined++;
              q = g95x_option.canonic ? st->q : p;
              
              
	      fprintf( out, REFFMT0, q );
	    break;
	    case 'A': 
	      p = va_arg( ap, g95x_pointer );
	      s = va_arg( ap, char* );
              st = g95x_get_bbt_id( p, s, &root_bbt_id );
              if( st == NULL )
                st = g95x_new_bbt_id( p, s, &root_bbt_id );
              if( ( FMT[j-5] == ' ' ) && ( FMT[j-4] == 'i' ) 
               && ( FMT[j-3] == 'd' ) && ( FMT[j-2] == '=' ) 
               && ( FMT[j-1] == '"' )
              ) st->defined++;
              q = g95x_option.canonic ? st->q : p;
              
              
	      fprintf( out, REFFMT0, q );
              
              if( ! g95x_option.canonic ) {
                int i;
                fprintf( out, "x" );
                for( i = 0; s[i]; i++ ) 
                  fprintf( out, "%2.2x", s[i] );
              }
              
	    break;
	    case 'T':
	      p = va_arg( ap, g95x_pointer );
	      fprintf( out, REFFMT1, p );
	    break;
	    default:
	      g95x_die( "Unknown format" );
	    break;
	  }
          i = j + 2;
	  continue;
        }
      }
      fprintf( out, FMT + i );
      break;
    }

    if( l >= 256 ) 
      g95_free( FMT );

    
    va_end( ap );
  }
/*
    va_start( ap, fmt );
    n += vfprintf( g95x_option.g95x_out, fmt, ap );
*/

  if( g95x_option.indent ) {
    for( i = 0; fmt[i]; i++ ) {
      if( ( fmt[i] == '<' ) && ( fmt[i+1] != '/' ) && ( fmt[i+1] != '!' ) ) indent++;
      if( ( fmt[i] == '/' ) && ( fmt[i+1] == '>' ) ) indent--;
    }
  }

  return n;
}

/*
 * Prints a symbol name, with the x.module:: prefix, if necessary
 * here, we need the namespace the symbol is relative to,
 * in order to guess the right x.module:: prefix
 */

static void print_symbol_name( g95_symbol *sym, g95_namespace *ns ) {
  if( sym->module ) {
    if( symbol_intrinsic( sym ) ) {
      g95x_die( "Unexpected intrinsic" );
    } else if( ! strcmp( sym->module, "(global)" ) ) {
    } else if( strcmp( sym->module, ns->proc_name->name ) ) {
      g95x_print( "%s::", sym->module );
    }
  }
  if( ( sym->attr.flavor == FL_BLOCK_DATA ) 
  && ( ! strcmp( sym->name, BLANK_BLOCK_DATA_NAME ) ) 
  ) { 
    g95x_print( "BLOCK_DATA_" );
    return;
  }

  g95x_print( "%s", sym->name );
}



static int valid_symtree( g95_symtree* st ) {
  g95_symbol* sym = st->n.sym;
  
  if( st->name[0] == '@' )
    return 0;
    
  if( sym->x.is_generic ) 
    if( sym->attr.proc != PROC_INTRINSIC ) 
      return 0;   
  
  return 1;
}







/*
 *
 */

static void* type_id( g95_typespec *ts ) {
  g95_integer_info *ip;
  g95_ff *fp;
  g95_logical_info *lp;
  static int character_info;
  static int procedure_info;
  
  switch( ts->type ) {
    case BT_INTEGER:
      for( ip = g95_integer_kinds; ip->kind != 0; ip++ ) 
        if( ip->kind == ts->kind )
          return ip;
    break;
    case BT_REAL:
      for( fp = g95_real_kinds; fp->kind != 0; fp++ ) 
        if( fp->kind == ts->kind )
          return fp;
    break;
    case BT_COMPLEX:
      for( fp = g95_real_kinds; fp->kind != 0; fp++ ) 
        if( fp->kind == ts->kind )
          return (char*)fp + 1;
    break;
    case BT_LOGICAL:
      for( lp = g95_logical_kinds; lp->kind != 0; lp++ ) 
        if( lp->kind == ts->kind )
          return lp;
    break;
    case BT_CHARACTER: 
      return &character_info;
    break;
    case BT_DERIVED:
      return (char*)ts->derived + 1;
    break;
    case BT_PROCEDURE:
      return &procedure_info;
    break;
    default:
    break;
  }

  return NULL;
}


static void print_typespec( g95_typespec *ts, const char* prefix ) {
  void *typeid = type_id( ts );
  
  if( ( ts->type != BT_PROCEDURE ) && typeid )
    g95x_print( " %stype=\"%R\"", prefix, typeid );
    
  switch( ts->type ) {
    case BT_CHARACTER: if( ts->cl ) {
      g95_charlen *cl = ts->cl;
      g95_expr *len;
      if( cl->x.alias ) cl = cl->x.alias;
      len = cl->length;
      if( len && len->x.full ) len = len->x.full;
      if( g95x_nodummy_expr( len ) )
        g95x_print( " %slength=\"%R\"", prefix, len );
      else if( ( ! len ) || cl->x.star )
        g95x_print( " %slength=\"*\"", prefix );
    }
    break;
    case BT_PROCEDURE:
      g95x_print( " %stype=\"%R\"", prefix, typeid );
      if( ts->interface ) 
        g95x_print( " %sinterface=\"%R\"", prefix, ts->interface );
    break;
    default:
    break;
  }
  if( ts->x.kind ) 
    g95x_print( " %skind=\"%R\"", prefix, ts->x.kind->kind );
}


static const char* sort_as( g95_array_spec *as ) {
  const char* p = NULL;
  switch( as->type ) {
    case AS_EXPLICIT:      p = "EXPLICIT";        break;
    case AS_ASSUMED_SHAPE: p = "ASSUMED_SHAPE";   break;
    case AS_DEFERRED:      p = "DEFERRED";        break;
    case AS_ASSUMED_SIZE:  p = "ASSUMED_SIZE";    break;
    default:
      g95x_die( "Unknown shape\n" );
  }
  return p;
}

static const char* sort_cas( g95_coarray_spec *cas ) {
  const char* p = NULL;
  switch( cas->type ) {
    case CAS_DEFERRED: p = "DEFERRED";        break;
    case CAS_ASSUMED:  p = "ASSUMED_SIZE";    break;
    default:
      g95x_die( "Unknown shape\n" );
  }
  return p;
}

static void print_array_spec( g95_array_spec *as, const char* prefix ) {
  int i;
  int constant = 1;
  
  
  if( ! as ) 
    return;
    
  if( as->x.alias ) 
    as = (g95_array_spec*)as->x.alias;
    
  g95x_print( " %sdimension=\"", prefix );
  for( i = 0; i < as->rank; i++ ) {
    g95_expr *lower = as->lower[i];
    g95_expr *upper = as->upper[i];
    
    if( lower && ( lower->type != EXPR_CONSTANT ) ) 
      constant = 0;
    if( ( ! upper ) || ( upper->type != EXPR_CONSTANT ) ) 
      constant = 0;
    
    if( lower && lower->x.full ) lower = lower->x.full;
    if( upper && upper->x.full ) upper = upper->x.full;
    
    if( g95x_nodummy_expr( lower ) )
      g95x_print( "%R", lower );
      
    g95x_print( ":" );
    
    if( upper )
      g95x_print( "%R", upper );
      
    if( ( as->type == AS_ASSUMED_SIZE ) 
     && ( i == ( as->rank - 1 ) ) 
     && ( ! upper ) ) {
      g95x_print( "*" );
    }


    if( i + 1 < as->rank ) 
      g95x_print( "," );
    
  }
  g95x_print( "\"" );
  
  g95x_print( " %sshape=\"%s\"", prefix, sort_as( as ) );
  

}

static void print_coarray_spec( g95_coarray_spec *cas, const char* prefix ) {
  int i;
  int constant = 1;
  
  
  if( ! cas ) 
    return;
    
  if( cas->x.alias ) 
    cas = (g95_coarray_spec*)cas->x.alias;
    
  g95x_print( " %scodimension=\"", prefix );
  for( i = 0; i < cas->rank; i++ ) {
    g95_expr *lower = cas->lower[i];
    g95_expr *upper = cas->upper[i];
    
    if( lower && ( lower->type != EXPR_CONSTANT ) ) 
      constant = 0;
    if( ( ! upper ) || ( upper->type != EXPR_CONSTANT ) ) 
      constant = 0;
    
    if( lower && lower->x.full ) lower = lower->x.full;
    if( upper && upper->x.full ) upper = upper->x.full;
    
    if( g95x_nodummy_expr( lower ) )
      g95x_print( "%R", lower );
      
    g95x_print( ":" );
    
    if( upper )
      g95x_print( "%R", upper );
      
    if( ( cas->type == CAS_ASSUMED ) 
     && ( i == ( cas->rank - 1 ) ) 
     && ( ! upper ) ) {
      g95x_print( "*" );
    }


    if( i + 1 < cas->rank ) 
      g95x_print( "," );
    
  }
  g95x_print( "\"" );
  
  g95x_print( " %scoshape=\"%s\"", prefix, sort_cas( cas ) );
  

}


static void print_ref( g95_ref *d ) { 
 
  g95x_print( "<ref id=\"%R\" type=\"%s\"", d, sort_ref( d ) );
 
  switch( d->type ) {  
    case REF_ARRAY: 
      if( d->u.ar.type != AR_FULL ) {
        int i;
        g95x_print( " dimension=\"" );
        for( i = 0; i < d->u.ar.dimen; i++ ) {
          switch( d->u.ar.dimen_type[i] ) {
            case DIMEN_RANGE:
              if( d->u.ar.start[i] ) 
                g95x_print( "%R", d->u.ar.start[i] );
              g95x_print( ":" );
              if( d->u.ar.end[i] ) 
                g95x_print( "%R", d->u.ar.end[i] );
              if( d->u.ar.stride[i] ) 
                g95x_print( ":%R", d->u.ar.stride[i] );
            break;
            case DIMEN_ELEMENT:
              if( d->u.ar.start[i] ) 
                g95x_print( "%R", d->u.ar.start[i] );
            break;
            case DIMEN_VECTOR:
              g95x_print( "%R", d->u.ar.start[i] );
            break;
            default:
              g95x_die( "Unexpected dimen_type\n" );
            break;
          }
          if( i < d->u.ar.dimen - 1 ) 
            g95x_print( "," );
        } 
        g95x_print( "\"" );
      } 
    break;

    case REF_COMPONENT: 
      g95x_print( 
        " component_type=\"%R\" component=\"%R\"", d->u.c.sym, d->u.c.component
      );
    break;      
    
    case REF_SUBSTRING: 
      g95x_print( " range=\"" );
      if( d->u.ss.start ) 
        g95x_print( "%R", d->u.ss.start );
      g95x_print( ":" );
      if( d->u.ss.end ) 
        g95x_print( "%R", d->u.ss.end );
      g95x_print( "\"" );
    break;     
    default: 
      g95x_die( "Unknown ref type for %R\n", d );
    break;       
  }  
  
  g95x_print( "/>\n" );    
}  

static void print_component( g95_component *c ) {    
    
  g95x_print( 
    "<component id=\"%R\" name=\"%s\"", c, c->name
  );
  
  if( c->initializer ) {
    g95_expr *init = c->initializer;
    if( init && init->x.full ) init = init->x.full;
    g95x_print( " initializer=\"%R\"", init );
  }
  if( c->pointer ) 
    g95x_print( " pointer=\"1\"" );
  if( c->allocatable ) 
    g95x_print( " pointer=\"1\"" );
  if( c->nopass ) 
    g95x_print( " nopass=\"1\"" );
  if( c->as )
    print_array_spec( c->as, "" );
  if( c->cas )
    print_coarray_spec( c->cas, "" );

  print_typespec( &c->ts, "" );
  
  
  g95x_print( "/>\n" );
}

static void print_attributes( symbol_attribute *a, const char* prefix ) {
  switch( a->access ) {
    case ACCESS_PUBLIC:  g95x_print( " %saccess=\"PUBLIC\"", prefix );  break;
    case ACCESS_PRIVATE: g95x_print( " %saccess=\"PRIVATE\"", prefix ); break;
    default:
    break;
  }
  
  switch( a->intent ) {
    case INTENT_IN:    g95x_print( " %sintent=\"IN\"", prefix );    break;
    case INTENT_OUT:   g95x_print( " %sintent=\"OUT\"", prefix );   break;
    case INTENT_INOUT: g95x_print( " %sintent=\"INOUT\"", prefix ); break;
    default:
    break;
  }
  
  switch( a->proc ) {
    case PROC_MODULE:      g95x_print( " %sproc=\"MODULE\"", prefix );      break;
    case PROC_INTERNAL:    g95x_print( " %sproc=\"INTERNAL\"", prefix );    break;
    case PROC_DUMMY:       g95x_print( " %sproc=\"DUMMY\"", prefix );       break;
    case PROC_INTRINSIC:   g95x_print( " %sproc=\"INTRINSIC\"", prefix );   break;
    case PROC_ST_FUNCTION: g95x_print( " %sproc=\"ST_FUNCTION\"", prefix ); break;
    case PROC_EXTERNAL:    g95x_print( " %sproc=\"EXTERNAL\"", prefix );    break;
    default:
    break;
  }
  
#define print_attr( attr ) if( a->attr ) \
  { g95x_print( " %s%s=\"%d\"", prefix, #attr, a->attr ); }
  
  print_attr( allocatable );
  print_attr( external );
  print_attr( intrinsic );
  print_attr( optional );
  print_attr( pointer );
  print_attr( save );
  print_attr( target );
  print_attr( dummy );     
  print_attr( result_var );
  print_attr( entry );
  print_attr( data );
  print_attr( equivalenced );
  print_attr( in_namelist );
  print_attr( in_common );
  print_attr( sequence );
  print_attr( elemental );
  print_attr( pure );
  print_attr( recursive );
  print_attr( protected );
  print_attr( value );
  if( a->volatile_ ) 
    g95x_print( " %svolatile=\"1\"", prefix );
  if( a->async ) 
    g95x_print( " %sasynchronous=\"1\"", prefix );
  

#undef print_attr
  
}

static void print_symbol( g95_symbol* sym, g95_namespace *ns ) {
  symbol_attribute *a = &sym->attr;
  
  
  if( symbol_intrinsic( sym ) ) {
    g95x_print( 
      "<symbol id=\"%R\" name=\"%s\" flavor=\"INTRINSIC\"/>\n", 
      g95x_get_intrinsic_by_symbol( sym ), 
      sym->name 
    );
    return;
  }
  
  g95x_print( 
    "<symbol id=\"%R\" flavor=\"%s\" name=\"", 
    sym, sort_flavor( &sym->attr, 1 ) 
  );
  
  print_symbol_name( sym, ns );

  g95x_print( "\"" );

  if( sym->x.imported ) {
    g95x_print( "/>\n" );
    return;
  }
  
  g95x_print( " namespace=\"%R\"", sym->ns );
  
  if( sym->formal_ns ) 
    g95x_print( " formal_namespace=\"%R\"", sym->formal_ns );
  


  print_attributes( &sym->attr, "" );



  if( a->bind ) {
    g95x_print( " bind=\"C\"" );
    if( sym->x.bind ) {
      g95_expr *xbind = sym->x.bind;
      if( xbind && xbind->x.full ) xbind = xbind->x.full;
    
      if( g95x_nodummy_expr( xbind ) )
        g95x_print( " c_name=\"%R\"", xbind );
    
      
    }
  }

  print_typespec( &sym->ts, "" );
  print_array_spec( sym->as, "" );
  print_coarray_spec( sym->cas, "" );



  switch( a->flavor ) {
    case FL_NAMELIST: 
      if( sym->namelist ) {
        g95_namelist *nam;
        g95x_print( " symbols=\"" );
        for( nam = sym->namelist; nam; nam = nam->next ) {
          g95x_print( "%R", nam->sym );
          if( nam->next ) g95x_print( "," );
        }
        g95x_print( "\"" );
      }
    break;
    case FL_DERIVED:
      if( sym->components ) {
        g95_component *cmp;
        g95x_print( " components=\"" );
        for( cmp = sym->components; cmp; cmp = cmp->next ) {
          g95x_print( "%R", cmp );
          if( cmp->next ) g95x_print( "," );
        }
        g95x_print( "\"" );
      }
    break;
    case FL_PROCEDURE:
      if( sym->formal ) {
        g95_formal_arglist *fal;
        g95x_print( " args=\"" );
        for( fal = sym->formal; fal; fal = fal->next ) {
          if( fal->sym )
            g95x_print( "%R", fal->sym );
          else 
            g95x_print( "*" );
          if( fal->next ) g95x_print( "," );
        }
        g95x_print( "\"" );
      }
      if( sym->result && ( sym->result != sym ) ) {
        g95x_print( " result=\"%R\"", sym->result );
      }
    break;
    default:
    break;
  }

  if( sym->value ) {
    g95_expr *value = sym->value;
    if( value && value->x.full ) value = value->x.full;
    
    if( g95x_nodummy_expr( value ) )
      g95x_print( " value=\"%R\"", value );
    
  }


  
  g95x_print( "/>\n" );
  
  
}



static void _print_symbol0( g95_symtree* st, void* data ) {
  g95_namespace* ns = data;
  g95x_traverse_callback xtc;
  _print_symbol_data psd;
  g95_symbol *sym = st->n.sym;
  


  if( ! g95x_symbol_valid( sym ) )
    return;

 
  if( valid_symtree( st ) && strcmp( st->name, sym->name ) ) 
    g95x_print( 
      "<symbol id=\"%A\" flavor=\"ALIAS\" name=\"%s\" target=\"%R\"/>\n", 
      sym, st->name, st->name, sym
    );
  

  psd.ns = ns;
  

  g95x_init_callback( &xtc );
  
  xtc.expr_callback        = _print_expr;
  xtc.constructor_callback = _print_constructor;
  xtc.symbol_callback      = _print_symbol;
  xtc.ref_callback         = _print_ref;
  xtc.component_callback   = _print_component;
  
  xtc.symbol_data          = &psd;
  
  g95x_traverse_symbol( sym, &xtc );
}

static void print_symbols( g95_namespace *ns ) {
  
  g95x_traverse_symtree( ns->sym_root, _print_symbol0, ns );
  
}



static void _print_symbol_list_item( g95_symtree* st, void* data ) {
  bbt_list_item_data *psd = data;
  g95_symbol* s = st->n.sym;
  
  
  if( ! g95x_symbol_valid( s ) )
    return;

  if( symbol_intrinsic( s ) )
    return;

 
  if( valid_symtree( st ) && strcmp( st->name, s->name ) ) {
    
    if( psd->printed )
      g95x_print( "," );

    g95x_print( "%A", s, st->name );
    psd->printed++;

  }

  if( g95x_get_bbt_id( (g95x_pointer)st->n.sym, NULL, &psd->bbt_root ) )
    return;
  else {

    if( psd->printed )
      g95x_print( "," );

    g95x_print( "%R", s );
    psd->printed++;

    g95x_new_bbt_id( (g95x_pointer)st->n.sym, NULL, &psd->bbt_root );
  }
}

static void print_symbol_list( g95_namespace *ns ) {
  bbt_list_item_data psd;
  

  psd.ns = ns;
  psd.printed = 0;
  psd.bbt_root.root = NULL;
  psd.bbt_root.bbt_id_number = 0;
  
  g95x_print( " symbols=\"" );
  g95x_traverse_symtree( ns->sym_root, _print_symbol_list_item, &psd );
  g95x_print( "\"" );
  
  g95x_free_bbt_ids( &psd.bbt_root );
  
  
}



/*
 * Commons
 */

static int valid_common( g95_common_head *c ) {
  g95_symbol *sym;

  /* Don't print commons imported from another module 
     It is forbidden to mix symbols from different units
     within the same common */
  sym = c->head; 

  if( sym && sym->x.imported )
    return 0;
  
  return 1;
}


static void _print_common( g95_symtree *st, void* data ) {
  g95_symbol *sym;
  g95_common_head *c = st->n.common;
  
  if( ! valid_common( c ) )
    return;
  
  /* a common is not associated w/ a symbol */
  g95x_print( 
    "<common id=\"%R\" name=\"%s\" symbols=\"", c, st->name 
  );
  for( sym = c->head; sym; sym = sym->common_next ) {
    /* a symbol within a common cannot belong to another x.module */
    g95x_print( "%R", sym );
    if( sym->common_next ) 
      g95x_print( "," ); 
  }
  g95x_print( "\"/>\n" );
  
}

static void print_commons( g95_namespace *ns ) {
  
  if( ( ! ns->common_root ) && ( ! ns->blank_common ) ) 
    return; 

  g95x_traverse_symtree( ns->common_root, _print_common, ns );

  if( ns->blank_common ) {
    g95_symbol *sym;
    g95x_print( 
      "<common id=\"%R\" blank=\"1\" symbols=\"", ns->blank_common
    );
    for( sym = ns->blank_common->head; sym; sym = sym->common_next ) {
      g95x_print( "%R", sym );
      if( sym->common_next ) g95x_print( "," ); 
    }
    g95x_print( "\"/>\n" );
  }

}


typedef struct _print_common_list_data {
  g95_namespace *ns;
  int printed;
} _print_common_list_data;

static void _print_common_list_item( g95_symtree *st, void* data ) {
  _print_common_list_data* psd = data;
  g95_common_head *c = st->n.common;
  
  
  if( ! valid_common( c ) )
    return;
  
  
  if( psd->printed )
    g95x_print( "," );
  g95x_print( "%R", c );
  psd->printed++;
  
  
  
}


static void print_common_list( g95_namespace *ns ) {
  _print_common_list_data psd;
  
  psd.ns = ns;
  psd.printed = 0;
  
  
  g95x_print( " commons=\"" );
  
  if( ( ! ns->common_root ) && ( ! ns->blank_common ) ) 
    goto end; 

  g95x_traverse_symtree( ns->common_root, _print_common_list_item, &psd );

  if( ns->blank_common ) {
    if( psd.printed )
      g95x_print( "," );
    g95x_print( "%R", ns->blank_common );
  }

end:
  g95x_print( "\"" );
}


/*
 * Equivalences
 */

static void print_equivs( g95_namespace *ns ) {
  g95x_traverse_callback xtc;
  g95_equiv *e, *f;
  g95x_init_callback( &xtc );
  
  xtc.expr_callback        = _print_expr;
  xtc.ref_callback         = _print_ref;
  xtc.constructor_callback = _print_constructor;
  
  if( ns->equiv == NULL )
    return;
    
  for( e = ns->equiv; e; e = e->next ) {
    if( e->x.imported )
      continue;
    g95x_print( "<equiv id=\"%R\" exprs=\"", e );
    for( f = e; f; f = f->eq ) {
      g95_expr *g = f->expr;
      if( g && g->x.full ) g = g->x.full;
      g95x_print( "%R", g );
      if( f->eq ) 
        g95x_print( "," );
    }
    g95x_print( "\"/>\n" );
    for( f = e; f; f = f->eq ) {
      g95_expr *g = f->expr;
      if( g && g->x.full ) g = g->x.full;
      g95x_traverse_expr( g, &xtc );
    }
  }
  
}

static void print_equiv_list( g95_namespace *ns ) {
  g95_equiv *e;
  int printed = 0;

  g95x_print( " equivs=\"" );
  
  for( e = ns->equiv; e; e = e->next ) {
    if( e->x.imported )
      continue;
    if( printed++ )
      g95x_print( "," );
    g95x_print( "%R", e );
    
  }

  g95x_print( "\"" );
}

/*
 * Generics
 */

static void print_generic( g95_symtree *st, g95_namespace *ns ) {
  g95_interface *intf;
  void *p;
  const char* name;
  const char* module;
  
  g95x_resolve_gen( st->name, ns->interface ? NULL : ns->parent, &p );
  
  g95x_print( "<generic id=\"%R\" base=\"%R\"", st, p );
    
  
  
  if( st->access != ACCESS_UNKNOWN ) {
    const char *p = NULL;
    switch( st->access ) {
      case ACCESS_PUBLIC:
        p = "PUBLIC";
      break; 
      case ACCESS_PRIVATE:
        p = "PRIVATE";
      break;
      default:
      break;
    } 
    g95x_print( " access=\"%s\"", p );
  }
  
  g95x_print( " procedures=\"" );

  for( intf = st->n.generic; intf; intf = intf->next ) {
    g95x_print( "%R", intf->sym );
    if( intf->next ) 
      g95x_print( "," );
  }
  
  g95x_print( "\"/>\n" );
  
  name = "@";
  module = "@";
  
#define compare_s( s, t ) ( ( ( s ) != NULL ) && ( ( t ) == NULL ) ) \
                      || ( ( ( s ) == NULL ) && ( ( t ) != NULL ) ) \
                      || ( ( ( s ) == NULL ) && ( ( t ) == NULL ) ? 0 : strcmp( ( s ), ( t ) ) )


  for( intf = st->n.generic; intf; intf = intf->next ) {
    if( compare_s( name, intf->x.name ) 
     || compare_s( module, intf->x.module ) 
    ) {
      if( intf->x.module && intf->x.name ) {
        int printed = 0;
        g95_interface *intf1;
        g95x_print( "<generic id=\"%R\" name=\"%s::%s\" procedures=\"", intf, intf->x.module, intf->x.name );
        for( intf1 = intf; intf1; intf1 = intf1->next ) {
          if( compare_s( intf1->x.name, intf->x.name ) 
           || compare_s( intf1->x.module, intf->x.module )
          ) break; 
          if( printed++ )
            g95x_print( "," );
          g95x_print( "%R", intf1->sym );
        }
        g95x_print( "\"/>\n" );
      }
      name = intf->x.name;
      module = intf->x.module;
    }
  }

#undef compare_s  
}

static void _print_generic( g95_symtree *st, void* data ) {
  g95_namespace *ns = (g95_namespace*)data;
  print_generic( st, ns );
}

static void print_generics( g95_namespace *ns ) {
  
  if( ns->generic_root == NULL )
    return;
  
  g95x_traverse_symtree( ns->generic_root, _print_generic, ns );

}




static void _print_generic_list_item( g95_symtree* st, void* data ) {
  bbt_list_item_data *psd = data;
  void *p = (void*)st;
  
  if( g95x_get_bbt_id( (g95x_pointer)p, NULL, &psd->bbt_root ) )
    return;
  else {
    if( psd->printed )
      g95x_print( "," );
    g95x_print( "%R", p );
    psd->printed++;
    g95x_new_bbt_id( (g95x_pointer)p, NULL, &psd->bbt_root );
  }
}

static void print_generic_list( g95_namespace *ns ) {
  bbt_list_item_data psd;
  

  psd.ns = ns;
  psd.printed = 0;
  psd.bbt_root.root = NULL;
  psd.bbt_root.bbt_id_number = 0;
  
  g95x_print( " generics=\"" );
  g95x_traverse_symtree( ns->generic_root, _print_generic_list_item, &psd );
  g95x_print( "\"" );
  
  g95x_free_bbt_ids( &psd.bbt_root );
  
  
}





/*
 * User ops
 */


static void print_user_operator( g95_symtree *st, g95_namespace *ns ) {
  g95_user_op *op = st->n.uop;
  g95_interface *intf;
  void* p;
  const char* name;
  const char* module;
  
  g95x_resolve_uop( st->name, ns->interface ? NULL : ns->parent, &p );
  
  g95x_print( 
    "<operator id=\"%R\" base=\"%R\"", st, p );
    
  g95x_print( " procedures=\"" );

  for( intf = op->operator; intf; intf = intf->next ) {
    g95x_print( "%R", intf->sym );
    if( intf->next ) 
      g95x_print( "," );
  }
  switch( op->access ) {
    case ACCESS_PUBLIC:
      g95x_print( " access=\"PUBLIC\"" );
    break;
    case ACCESS_PRIVATE:
      g95x_print( " access=\"PRIVATE\"" );
    break;
    default: break;
  }
  g95x_print( "\"/>\n" );

  name = "@";
  module = "@";
  
#define compare_s( s, t ) ( ( ( s ) != NULL ) && ( ( t ) == NULL ) ) \
                      || ( ( ( s ) == NULL ) && ( ( t ) != NULL ) ) \
                      || ( ( ( s ) == NULL ) && ( ( t ) == NULL ) ? 0 : strcmp( ( s ), ( t ) ) )


  for( intf = st->n.uop->operator; intf; intf = intf->next ) {
    if( compare_s( name, intf->x.name ) 
     || compare_s( module, intf->x.module ) 
    ) {
      if( intf->x.module && intf->x.name ) {
        int printed = 0;
        g95_interface *intf1;
        g95x_print( "<operator id=\"%R\" name=\"%s::%s\" procedures=\"", intf, intf->x.module, intf->x.name );
        for( intf1 = intf; intf1; intf1 = intf1->next ) {
          if( compare_s( intf1->x.name, intf->x.name ) 
           || compare_s( intf1->x.module, intf->x.module )
          ) break; 
          if( printed++ )
            g95x_print( "," );
          g95x_print( "%R", intf1->sym );
        }
        g95x_print( "\"/>\n" );
      }
      name = intf->x.name;
      module = intf->x.module;
    }
  }

#undef compare_s  


}

static void _print_user_operator( g95_symtree *st, void* data ) {
  g95_namespace *ns = (g95_namespace*)data;
  print_user_operator( st, ns );
}

static void print_user_operators( g95_namespace *ns ) {
  
  if( ns->uop_root == NULL )
    return;
  
  g95x_traverse_symtree( ns->uop_root, _print_user_operator, ns );

}



static void _print_user_operator_list_item( g95_symtree* st, void* data ) {
  bbt_list_item_data *psd = data;
  void *p = st;
  
  if( g95x_get_bbt_id( (g95x_pointer)p, NULL, &psd->bbt_root ) )
    return;
  else {
    if( psd->printed )
      g95x_print( "," );
    g95x_print( "%R", p );
    psd->printed++;
    g95x_new_bbt_id( (g95x_pointer)p, NULL, &psd->bbt_root );
  }
}

static void print_user_operator_list( g95_namespace *ns ) {
  bbt_list_item_data psd;
  

  psd.ns = ns;
  psd.printed = 0;
  psd.bbt_root.root = NULL;
  psd.bbt_root.bbt_id_number = 0;
  
  g95x_print( " user_operators=\"" );
  g95x_traverse_symtree( ns->uop_root, _print_user_operator_list_item, &psd );
  g95x_print( "\"" );
  
  g95x_free_bbt_ids( &psd.bbt_root );
  
  
}


/*
 * datas
 */

static void print_data_var( g95_data_variable *var ) {
  g95x_print( "<data_var id=\"%R\"", var );
  if( var->expr ) {
    g95x_print( " expr=\"%R\"", var->expr );
  }
  if( var->list ) {
    g95_data_variable *v;
    g95x_print( " list_vars=\"" );
    for( v = var->list; v; v = v->next ) {
      g95x_print( "%R", v );
      if( v->next ) 
        g95x_print( "," );
    }
    g95x_print( "\"" );    
    g95x_print( " iterator=\"" );
    g95x_print( 
      "%R:%R:%R", var->iter.var, var->iter.start->x.full, var->iter.end->x.full
    );
    if( g95x_nodummy_expr( var->iter.step->x.full ) )
      g95x_print( ":%R", var->iter.step->x.full );
    g95x_print( "\"" );
  }
  g95x_print( "/>\n" );
}

static void print_datas( g95_namespace *ns ) {
  g95x_traverse_callback xtc;
  g95_data *d;
  
  g95x_init_callback( &xtc );
  xtc.expr_callback = _print_expr;
  xtc.ref_callback = _print_ref;
  xtc.constructor_callback = _print_constructor;
  xtc.data_var_callback = _print_data_var;
  
  if( ns->data == NULL ) 
    return;
  
  for( d = ns->data; d; d = d->next ) {
    g95_data_value *val;
    g95_data_variable *var;

    g95x_print( "<data id=\"%R\" vars=\"", d );
    for( var = d->var; var; var = var->next ) {
      g95x_print( "%R", var );
      if( var->next ) 
        g95x_print( "," );
    }
    
    g95x_print( "\" values=\"" );
    for( val = d->value; val; val = val->next ) {
      g95_expr *v = val->expr, *r = val->x.repeat;
      if( v && v->x.full ) v = v->x.full;
      if( r && r->x.full ) r = r->x.full;
      if( r ) 
        g95x_print( "%R*", r );
      g95x_print( "%R", v );
      if( val->next ) 
        g95x_print( "," );
    }
    g95x_print( "\"" );
    
    
    g95x_print( "/>\n" );
    
    for( var = d->var; var; var = var->next ) {
      g95x_traverse_data_var( var, &xtc );
    }

    for( val = d->value; val; val = val->next ) {
      g95_expr *v = val->expr, *r = val->x.repeat;
      if( v && v->x.full ) v = v->x.full;
      if( r && r->x.full ) r = r->x.full;
      g95x_traverse_expr( r, &xtc );
      g95x_traverse_expr( v, &xtc );
    }
    
    
  }
  
}


static void print_data_list( g95_namespace *ns ) {
  g95_data *d;
  int printed = 0;
  
  g95x_print( " datas=\"" );
  for( d = ns->data; d; d = d->next ) {
    if( printed ) 
      g95x_print( "," );
    g95x_print( "%R", d );
    printed++;
  }
  g95x_print( "\"" );
  
}



/*
 * Intrinsic ops
 */

static void print_intrinsic_operators( g95_namespace *ns ) {
  g95_interface *intf;
  int l;
  g95_namespace* nsp;
  
  nsp = ns->interface ? NULL : ns->parent;

  for( l = 0; l < G95_INTRINSIC_OPS; l++ ) {
    if( ns->operator[l] != NULL ) {
      void* p;
      g95x_resolve_iop( l, nsp, &p );
      g95x_print( 
        "<operator id=\"%R\" base=\"%R\" procedures=\"", 
        ns->operator[l], p 
      );
      for( intf = ns->operator[l]; intf; intf = intf->next ) {
        g95x_print( "%R", intf->sym );
        if( intf->next ) 
          g95x_print( "," );
      }
      g95x_print( "\"/>\n" );
    }
  }
}


static void print_intrinsic_operator_list( g95_namespace *ns ) {
  int l;
  int printed = 0;
  
  g95x_print( " intrinsic_operators=\"" );
  
  for( l = 0; l < G95_INTRINSIC_OPS; l++ ) 
    if( ns->operator[l] != NULL ) {
      if( printed )
        g95x_print( "," );
      g95x_print( "%R", ns->operator[l] );
      printed++;
    }

  g95x_print( "\"" );
}




/*
 * 
 */


static void print_implicit_list( g95_typespec *ts ) {
  int l;
  void* typeid;
  int printed = 0;

  g95x_print( " implicit=\"[" );
  
  for( l = 0; l < G95_LETTERS; l++ ) {  
    typeid = type_id( &ts[l] ); 
    if( typeid ) {
      if( printed )
        g95x_print( "," );
      g95x_print( "['%c','%R'", 'a' + l, typeid );

      if( ts[l].cl ) {
        g95_expr *length = ts[l].cl->length;
        if( length->x.full ) length = length->x.full;
        g95x_print( ",'%R'", length );
      } else
        g95x_print( ",''" );

      if( ts[l].x.kind ) 
        g95x_print( ",'%R'", ts[l].x.kind->kind );
      else
        g95x_print( ",''" );

      g95x_print( "]" );
      printed++;
    }
  }

  g95x_print( "]\"" );

}

static void print_implicits( g95_typespec *ts ) {
  int l;
  g95x_traverse_callback xtc;
  void* typeid;
  g95x_init_callback( &xtc );
  
  xtc.expr_callback        = _print_expr;
  xtc.constructor_callback = _print_constructor;
  xtc.ref_callback         = _print_ref;
  
    

  for( l = 0; l < G95_LETTERS; l++ ) {  
    typeid = type_id( &ts[l] ); 
    if( typeid ) {
      
      if( ts[l].x.kind ) {
        g95x_traverse_expr( ts[l].x.kind->kind, &xtc );
      }
      if( ts[l].cl ) {
        g95_expr *length = ts[l].cl->length;
        if( length->x.full ) length = length->x.full;
        g95x_traverse_expr( length, &xtc );
      }
    }
  }

}


static void print_labels( g95_namespace *ns  ) {
  g95_st_label *label;
  
  for( label = ns->st_labels; label; label = label->next ) {
    g95x_print( "<label id=\"%R\" value=\"%d\"", 
      label, label->value );
    if( ! g95x_option.code_only ) {
      g95x_print( " f=\"%R\" loc=\"[%d,%d,%d,%d]\"", 
        label->x.where.lb1->file, 
        label->x.where.lb1->linenum - 1, 
        label->x.where.column1 - 1,  
        label->x.where.lb2->linenum - 1, 
        label->x.where.column2 - 1  
      );
    }
    g95x_print( "/>\n" );
    
  }
}

static void print_label_list( g95_namespace *ns  ) {
  g95_st_label *label;
  int printed = 0;
  
  g95x_print( " labels=\"" );
  for( label = ns->st_labels; label; label = label->next ) {
    if( printed )
      g95x_print( "," );
    g95x_print( "%R", label );
    printed++;
  }
  g95x_print( "\"" );
}


/* Prints out the current namespace */

static int valid_namespace( g95_namespace *ns ) {
  /* blockdata can be anonymous */
  if( ( ns->proc_name == NULL ) 
   && ( ns->state != COMP_BLOCK_DATA )
  ) return 0; 
  return 1;
}

static void print_contained_ns_list( g95_namespace *ns ) {
  g95_namespace *ns1;
  int printed = 0;
  g95x_print( " contained_namespaces=\"" );
  for( ns1 = ns->contained; ns1; ns1 = ns1->sibling ) {
    if( ! valid_namespace( ns1 ) )
      continue;
    if( printed )
      g95x_print( "," );
    g95x_print( "%R", ns1 );
    printed++;
  }
  g95x_print( "\"" );
}

static void print_interface_ns_list( g95_namespace *ns ) {
  int printed = 0;
  g95x_statement *sta;
  g95x_symbol_list *sl;
  
  g95x_print( " interface_namespaces=\"" );
  for( sta = ns->x.statement_head; ; sta = sta->next ) {
    if( sta->type == ST_INTERFACE ) {
      for( sl = sta->u.interface.symbol_head; sl; sl = sl->next ) {
        if( printed )
          g95x_print( "," );
        g95x_print( "%R", sl->symbol->formal_ns );
        printed++;
      }
    }
    if( sta == ns->x.statement_tail )
      break;
  }
  g95x_print( "\"" );
}

static void print_contained_ns( g95_namespace *ns ) {
  for( ns = ns->contained; ns; ns = ns->sibling ) 
    g95x_print_namespace( ns );
}

static void print_interface_ns( g95_namespace *ns ) {
  g95x_statement *sta;
  g95x_symbol_list *sl;
  
  for( sta = ns->x.statement_head; ; sta = sta->next ) {
    if( sta->type == ST_INTERFACE ) {
      for( sl = sta->u.interface.symbol_head; sl; sl = sl->next ) 
        g95x_print_namespace( sl->symbol->formal_ns );
    }
    if( sta == ns->x.statement_tail )
      break;
  }
}



void g95x_print_namespace( g95_namespace *ns ) {


  if( ! valid_namespace( ns ) )
    return;

  g95x_print( "<namespace id=\"%R\"", ns ); 
  
  if( ns->parent )
    g95x_print( " parent=\"%R\"", ns->parent );
  if( ns->proc_name )
    g95x_print( " symbol=\"%R\"", ns->proc_name );
  if( ns->interface ) 
    g95x_print( " interface=\"1\"" );
  if( ns->import ) 
    g95x_print( " import=\"1\"" );

  g95x_print( 
    " statement_head=\"%R\" statement_tail=\"%R\"", 
    ns->x.statement_head, ns->x.statement_tail 
  );


  print_implicit_list( &ns->default_type[0] );

  print_symbol_list( ns );
  
  print_common_list( ns );
  
  print_equiv_list( ns );
  
  print_generic_list( ns );
  
  print_user_operator_list( ns );
  
  print_intrinsic_operator_list( ns );
  
  print_data_list( ns );
  
  print_label_list( ns );

  print_contained_ns_list( ns );

  print_interface_ns_list( ns );

  g95x_print( "/>\n" );

  print_implicits( &ns->default_type[0] );

  print_symbols( ns );
  
  print_commons( ns );
  
  print_equivs( ns );
  
  print_generics( ns );
  
  print_user_operators( ns );

  print_intrinsic_operators( ns );
  
  print_datas( ns );

  print_labels( ns );


  
  print_contained_ns( ns );

  print_interface_ns( ns );
  
  
}

void g95x_print_statements() {
  g95x_statement *sta;
  for( sta = g95x_statement_head; sta; sta = sta->a_next ) 
    print_statement( sta );
 
}

static void print_io_var( g95_code *c ) {
  g95_code *d;

  g95x_print( "<io_var id=\"%R\" list_vars=\"", c->block );

  for( d = c->block; d; d = d->next ) {
    switch( d->type ) {
      case EXEC_DO:
        g95x_print( "%R", d->block );
      break;
      case EXEC_TRANSFER:
        g95x_print( "%R", d->expr );
      break;
      default:
      break;
    }
    if( d->next ) 
      g95x_print( "," );
  }
  g95x_print( "\" iterator=\"" );

  g95x_print( "%R", c->ext.iterator->var );   
  g95x_print( ":%R", c->ext.iterator->start );        
  g95x_print( ":%R", c->ext.iterator->end );
  if( g95x_nodummy_expr( c->ext.iterator->step ) )   
    g95x_print( ":%R", c->ext.iterator->step ); 
           
  g95x_print( "\"/>\n" );
  
  
}

g95x_voidp_list* set_voidp_prev( g95x_voidp_list* f ) {
  if( f == NULL )
    return NULL;
  for( ; f->next; f = f->next ) {
    f->next->prev = f;
  }
  return f;
}


/* Prints out a statement */

static void print_ST_END_INTERFACE( g95x_statement *sta ) {
  g95x_statement *sta1 = sta->u.interface.st_interface ? sta->u.interface.st_interface : sta;

  switch( sta1->u.interface.type ) {
    case INTERFACE_ABSTRACT:
    case INTERFACE_NAMELESS: 
    break;
    case INTERFACE_GENERIC:
      g95x_print( " block=\"%R\"", sta1->ext_locs[0].u.gnrc.intf );
    break;
    case INTERFACE_INTRINSIC_OP:
      g95x_print( " block=\"%R\"", sta1->ext_locs[0].u.inop.intf );
    break;
    case INTERFACE_USER_OP:
      g95x_print( " block=\"%R\"", sta1->ext_locs[0].u.usop.uop );
    break;
    default:
      g95x_die( "Unexpected interface" );
  }

}

static void print_ST_INTERFACE( g95x_statement *sta ) {
  g95x_symbol_list *sl;

  g95x_print( " interface_type=\"%s\"", sort_interface( sta->u.interface.type ) );
  
  switch( sta->u.interface.type ) {
    case INTERFACE_ABSTRACT:
    case INTERFACE_NAMELESS: 
    break;
    case INTERFACE_GENERIC:
      g95x_print( " generic=\"%R\"", sta->ext_locs[0].u.gnrc.intf );
    break;
    case INTERFACE_INTRINSIC_OP:
     g95x_print( " op=\"%R\"", sta->ext_locs[0].u.inop.intf );
    break;
    case INTERFACE_USER_OP:
      g95x_print( " op=\"%R\"", sta->ext_locs[0].u.usop.uop );
    break;
    default:
      g95x_die( "Unexpected interface" );
  }
  
  
  g95x_print( " namespaces=\"" );
  for( sl = sta->u.interface.symbol_head; sl; sl = sl->next ) {
    g95x_print( "%R", sl->symbol->formal_ns );
    if( sl->next ) g95x_print( "," );
  }
  g95x_print( "\"" );
  
  print_ST_END_INTERFACE( sta );
}


static void print_ST_FORMAT( g95x_statement *sta ) {
  g95x_print( 
     " here=\"%R\" expr=\"%R\"", 
     sta->u.format.here, sta->u.format.here->format
  );
}

static void print_ST_ENUM( g95x_statement *sta ) {
  if( sta->enum_bindc )
    g95x_print( " bind=\"C\"" );
}

static void print_ST_IMPLICIT( g95x_statement *sta ) {
  print_implicit_list( &sta->u.implicit.ts[0] );
}

static void print_ST_DATA_DECL( g95x_statement *sta ) {
  int n;
  const char* flavor = sort_flavor( &sta->u.data_decl.attr, 0 );
  g95x_voidp_list *f, *f0;

  if( flavor )
    g95x_print( " decl_flavor=\"%s\"", flavor );
    
  print_typespec( &sta->u.data_decl.ts, "decl_" );
  print_attributes( &sta->u.data_decl.attr, "decl_" );
  
  f0 = set_voidp_prev( sta->u.data_decl.decl_list );
  if( sta->in_derived )
    g95x_print( " decl_components=\"" );
  else
    g95x_print( " decl_symbols=\"" );

  for( f = f0; f; f = f->prev ) {
    if( f->type == G95X_VOIDP_SYMBOL ) {
      g95_symbol *sym = f->u.symbol;
      void* p;
      if( symbol_intrinsic( sym ) )
        p = g95x_get_intrinsic_by_symbol( sym );
      else                             
        p = sym;
      g95x_print( "%R", p );
    } else
      g95x_print( "%R", f->u.voidp );
    if( f->prev )
      g95x_print( "," );
  } 
  g95x_print( "\"" );

  if( ! sta->in_derived ) {

    n = 0;
    g95x_print( " decl_symbol_dimensions=\"" );
    for( f = f0; f; f = f->prev ) 
      if( f->dimension )
        n++;

    for( f = f0; f; f = f->prev ) {
      if( f->dimension ) {
        g95x_print( "%R", f->u.voidp );
        if( --n )
          g95x_print( "," );
      }
    } 
    g95x_print( "\"" );

    n = 0;
    g95x_print( " decl_symbol_initializations=\"" );
    for( f = f0; f; f = f->prev ) 
      if( f->init )
        n++;

    for( f = f0; f; f = f->prev ) {
      if( f->init ) {
        g95x_print( "%R", f->u.voidp );
        if( --n )
          g95x_print( "," );
      }
    } 
    g95x_print( "\"" );

  }

  if( sta->u.data_decl.as )
    print_array_spec( (g95_array_spec*)sta->u.data_decl.as, "decl_" );
  if( sta->u.data_decl.cas )
    print_coarray_spec( (g95_coarray_spec*)sta->u.data_decl.cas, "decl_" );
  
}


static void print_ST_ATTR_DECL( g95x_statement *sta ) {
  int printed = 0;
  g95_namespace *ns = sta->ns;
  symbol_attribute *attr = &sta->u.data_decl.attr;

  g95x_voidp_list *sl0 = set_voidp_prev( sta->u.data_decl.decl_list ), *sl;
  g95x_print( " attr_symbols=\"" );
  for( sl = sl0; sl; sl = sl->prev ) {
    void* p;

    if( sl->type == G95X_VOIDP_INTROP_IDX ) {
      p = ns->operator[sl->u.introp_idx];
      if( ! p ) 
        g95x_resolve_iop( sl->u.introp_idx, ns, &p );
    } else if( sl->type == G95X_VOIDP_SYMBOL ) {
      g95_symbol *sym = sl->u.symbol;
      if( symbol_intrinsic( sym ) ) {

        if( attr->intrinsic ) 
          p = g95x_get_intrinsic_by_symbol( sym );
        else {
          g95x_resolve_gen( sym->name, ns, &p );
          if( p == NULL )
            g95x_die( "Expected generic" ); 
        }
          
      } else if( sym->x.is_generic ) {
        g95x_resolve_gen( sym->name, ns, &p );
      } else                       
        p = sym;
    } else
      p = sl->u.voidp;
    
    if( p ) {
      if( printed )
        g95x_print( "," );
      g95x_print( "%R", p );
      printed++;
    } else {
      g95x_die( "" );
    }
  } 
  g95x_print( "\"" );
  
  if( attr->allocatable || attr->pointer || attr->dimension ) {
    printed = 0;
    g95x_print( " decl_symbol_dimensions=\"" );
    for( sl = sl0; sl; sl = sl->prev ) {
      if( sl->dimension ) {
        if( printed ) 
          g95x_print( "," );
        g95x_print( "%R", sl->u.voidp );
        printed++;
      }
    }
    g95x_print( "\"" );
  }
}

static void print_ST_PARAMETER( g95x_statement *sta ) {
  g95x_voidp_list *sl = set_voidp_prev( sta->u.data_decl.decl_list );;
  g95x_print( " decl_symbols=\"" );
  for( ; sl; sl = sl->prev ) {
    g95x_print( "%R", sl->u.voidp );
    if( sl->prev )
      g95x_print( "," );
  } 
  g95x_print( "\"" );
}

static void print_ST_COMMON( g95x_statement *sta ) {
  g95x_voidp_list *sl = set_voidp_prev( sta->u.common.common_list );
  int n;
  
  g95x_print( " decl_symbols=\"" );
  for( ; sl; sl = sl->prev ) {
    g95x_print( "%R", sl->u.voidp );
    if( sl->prev )
      g95x_print( "," );
  } 
  g95x_print( "\"" );


  sl = set_voidp_prev( sta->u.common.common_list );
  n = 0;
  g95x_print( " decl_symbol_dimensions=\"" );
  for( ; sl; sl = sl->prev ) {
    if( sl->dimension )
      n++;
  } 
  sl = set_voidp_prev( sta->u.common.common_list );
  for( ; sl; sl = sl->prev ) {
    if( sl->dimension ) {
      g95x_print( "%R", sl->u.voidp );
      if( --n )
        g95x_print( "," );
    }
  } 
  g95x_print( "\"" );

}

static void print_ST_DERIVED_DECL( g95x_statement *sta ) {
  print_attributes( &sta->u.data_decl.attr, "decl_" );
}


static void print_ST_SUBROUTINE( g95x_statement *sta ) {
  print_attributes( &sta->u.subroutine.attr, "decl_" );
}

static void print_ST_FUNCTION( g95x_statement *sta ) {
  print_typespec( &sta->u.function.ts, "decl_" );
  print_attributes( &sta->u.function.attr, "decl_" );
}

static void print_ST_DATA( g95x_statement *sta ) {
  g95_data* d;
  /* forward */
  for( d = sta->u.data.d1; ; d = d->next ) {
    if( d->next )
      d->next->x.prev = d;
    if( d->next == sta->u.data.d2 )
      break;
  }
  /* now backward */
  g95x_print( " datas=\"" );
  for( ; ; d = d->x.prev ) {
    g95x_print( "%R", d );
    if( d == sta->u.data.d1 )
      break;
    g95x_print( "," );
  }
  g95x_print( "\"" );
}
  
static void print_ST_EQUIVALENCE( g95x_statement *sta ) {
  g95_equiv* e;
  /* forward */
  for( e = sta->u.equiv.e1; ; e = e->next ) {
    if( e->next )
      e->next->x.prev = e;
    if( e->next == sta->u.equiv.e2 )
      break;
  }
  /* now backward */
  g95x_print( " equivs=\"" );
  for( ; ; e = e->x.prev ) {
    g95x_print( "%R", e );
    if( e == sta->u.equiv.e1 )
      break;
    g95x_print( "," );
  }
  g95x_print( "\"" );
}



static void print_ST_USE( g95x_statement *sta ) {
  int seen_module = 0;
  int i;
  g95_namespace *ns = sta->ns;
  g95x_ext_locus *xw;
  int printed = 0;
  
  if( sta->u.use.only ) 
    g95x_print( " only=\"1\"" );
    
  if( sta->n_ext_locs ) {
    for( i = 0; i < sta->n_ext_locs; i++ ) {
      xw = &sta->ext_locs[i];
      
      if( xw->type != G95X_LOCUS_LANG ) {
        if( seen_module ) {


          switch( xw->type ) {
            case G95X_LOCUS_GNRC:
              if( printed )
                g95x_print( "," );
              g95x_print( "%R", xw->u.gnrc.intf );
            break;
            case G95X_LOCUS_SMBL: 
              if( printed )
                g95x_print( "," );
              print_symbol_alias( &xw->loc, NULL, xw->u.smbl.symbol, NULL );
            break;
            case G95X_LOCUS_ASSC: 
              if( xw->u.assc.symbol )
                g95x_print( "=%R", xw->u.assc.symbol );
              else if( xw->u.assc.intf )
                g95x_print( "=%R", xw->u.assc.intf );
              else
                g95x_die( "" );
            break;
            case G95X_LOCUS_USOP:
              if( printed )
                g95x_print( "," );
              g95x_print( "%R", xw->u.usop.uop );
            break;
            case G95X_LOCUS_INOP: 
              if( printed )
                g95x_print( "," );
              g95x_print( "%R", xw->u.inop.intf );
            break;
            default:
              g95x_die( "Unexpected locus\n" );
            break;
          }
          
          printed++;
            
        } else {
          g95x_print( " module=\"" );
          print_ext_loc_name( xw, ns );
          g95x_print( "\" args=\"" );
          seen_module = 1;
        }
      }
    }
    g95x_print( "\"" );
  }
}



static void print_ST_ENUMERATOR( g95x_statement *sta ) {
  g95x_voidp_list *sl = set_voidp_prev( sta->u.enumerator.enumerator_list );
  
  g95x_print( " enums=\"" );
  for( ; sl; sl = sl->prev ) {
    g95x_print( "%R", sl->u.voidp );
    if( sl->prev )
      g95x_print( "," );
  } 
  g95x_print( "\"" );
}
  

  
static void print_ST_MODULE_PROC( g95x_statement *sta ) {
  int i;
  g95_namespace *ns = sta->ns;
  if( sta->n_ext_locs ) {
    g95x_print( " procedures=\"" );
    for( i = 0; i < sta->n_ext_locs; i++ ) {
      if( sta->ext_locs[i].type == G95X_LOCUS_SMBL ) {
        print_ext_loc_name( sta->ext_locs + i, ns );
        if( i < sta->n_ext_locs - 1 )
          g95x_print( "," );
      }
    }
    g95x_print( "\"" );
  }
}

  
static void print_ST_NAMELIST( g95x_statement *sta ) {
  int i;
  g95_namespace *ns = sta->ns;
  if( sta->n_ext_locs ) {
    g95x_print( " decl_symbols=\"" );
    for( i = 0; i < sta->n_ext_locs; i++ ) {
      if( sta->ext_locs[i].type == G95X_LOCUS_SMBL ) {
        print_ext_loc_name( sta->ext_locs + i, ns );
        if( i < sta->n_ext_locs - 1 )
          g95x_print( "," );
      }
    }
    g95x_print( "\"" );
  }
}

static void print_ST_STATEMENT_FUNCTION( g95x_statement *sta ) {
  int i;
  g95_namespace *ns = sta->ns;
  if( sta->n_ext_locs ) {
    g95x_print( " block=\"" );
    for( i = 0; i < sta->n_ext_locs; i++ ) {
      if( sta->ext_locs[i].type == G95X_LOCUS_SMBL ) {
        print_ext_loc_name( sta->ext_locs + i, ns );
        break;
      }
    }
    g95x_print( "\"" );
  }
}



  
static void print_ST_IMPORT( g95x_statement *sta ) {
  int i;
  g95_namespace *ns = sta->ns;
  if( sta->n_ext_locs ) {
    g95x_print( " decl_symbols=\"" );
    for( i = 0; i < sta->n_ext_locs; i++ ) {
      if( sta->ext_locs[i].type == G95X_LOCUS_SMBL ) {
        print_ext_loc_name( sta->ext_locs + i, ns );
        if( i < sta->n_ext_locs - 1 )
          g95x_print( "," );
      }
    }
    g95x_print( "\"" );
  }
}


  
static void print_ST_ENTRY( g95x_statement *sta ) {
  int i;
  g95_namespace *ns = sta->ns;
  if( sta->n_ext_locs ) {
    g95x_print( " block=\"" );
    for( i = 0; i < sta->n_ext_locs; i++ ) {
      if( sta->ext_locs[i].type == G95X_LOCUS_SMBL ) {
        print_ext_loc_name( sta->ext_locs + i, ns );
        break;
      }
    }
    g95x_print( "\"" );
  }
}


static void print_statement_parts_ST_FORMAT( g95x_statement *sta ) {
  g95x_traverse_callback xtc;
  g95x_init_callback( &xtc );
  
  xtc.expr_callback        = _print_expr;
  xtc.constructor_callback = _print_constructor;
  xtc.ref_callback         = _print_ref;

  g95x_traverse_expr( sta->u.format.here->format, &xtc );
}

static void print_statement_parts_ST_DATA_DECL( g95x_statement *sta ) {
  if( sta->u.data_decl.as ) {
    g95x_traverse_callback xtc;
    g95x_init_callback( &xtc );
  
    xtc.expr_callback        = _print_expr;
    xtc.constructor_callback = _print_constructor;
    xtc.ref_callback         = _print_ref;

    g95x_traverse_array_spec( (g95_array_spec*)sta->u.data_decl.as, &xtc );
  }
  if( sta->u.data_decl.ts.cl ) {
    g95_expr *e = sta->u.data_decl.ts.cl->length;
    
    g95x_traverse_callback xtc;
    g95x_init_callback( &xtc );
  
    xtc.expr_callback        = _print_expr;
    xtc.constructor_callback = _print_constructor;
    xtc.ref_callback         = _print_ref;
    
    if( e ) {
      e = e->x.full ? e->x.full : e;
      if( g95x_nodummy_expr( e ) )
        g95x_traverse_expr( e, &xtc );
    }
  }
}
  
int g95x_print_statement_ext_locs( g95x_statement* sta ) {
  int i;
  g95_namespace *ns = sta->ns;
  for( i = 0; i < sta->n_ext_locs; i++ ) {
    print_ext_locs( &sta->ext_locs[i], ns, NULL );
    if( ( i + 1 ) < sta->n_ext_locs ) 
      g95x_print( "," );
  }
  return sta->n_ext_locs;
}


static void print_statement( g95x_statement *sta ) {
  g95_namespace *ns = sta->ns;
  
  g95x_print( 
    "<statement id=\"%R\" type=\"%s\" namespace=\"%R\"", 
    sta, sort_statement( sta ), ns 
  );

  if( ! g95x_option.code_only ) {
    g95x_print( " f=\"%R\"", sta->where.lb1->file );
    g95x_print( " loc=\"" );
    print_locs( &sta->where );
    g95x_print( "\"" ); 
  }


  if( sta->here ) 
    g95x_print( " here=\"%R\"", sta->here );

  if( sta->a_next ) 
    g95x_print( " next=\"%R\"", sta->a_next );

  if( sta->a_prev ) 
    g95x_print( " prev=\"%R\"", sta->a_prev );

  if( sta->eblock )
    g95x_print( " eblock=\"%R\"", sta->eblock );

  if( sta->fblock )
    g95x_print( " fblock=\"%R\"", sta->fblock );

  if( sta->block )
    g95x_print( " block=\"%R\"", sta->block );
  
  if( sta->implied_enddo ) 
    g95x_print( " enddo=\"%d\"", sta->implied_enddo ); 

  switch( sta->type ) {
    case ST_INTERFACE:             print_ST_INTERFACE( sta );                break;
    case ST_END_INTERFACE:         print_ST_END_INTERFACE( sta );            break;
    case ST_FORMAT:                print_ST_FORMAT( sta );                   break;
    case ST_ENUM:                  print_ST_ENUM( sta );                     break;
    case ST_IMPLICIT:              print_ST_IMPLICIT( sta );                 break;
    case ST_DATA_DECL:             print_ST_DATA_DECL( sta );                break;
    case ST_ATTR_DECL:             print_ST_ATTR_DECL( sta );                break;
    case ST_PARAMETER:             print_ST_PARAMETER( sta );                break;
    case ST_COMMON:                print_ST_COMMON( sta );                   break;
    case ST_DERIVED_DECL:          print_ST_DERIVED_DECL( sta );             break;
    case ST_SUBROUTINE:            print_ST_SUBROUTINE( sta );               break;
    case ST_FUNCTION:              print_ST_FUNCTION( sta );                 break;
    case ST_DATA:                  print_ST_DATA( sta );                     break;
    case ST_EQUIVALENCE:           print_ST_EQUIVALENCE( sta );              break;
    case ST_USE:                   print_ST_USE( sta );                      break;
    case ST_ENUMERATOR:            print_ST_ENUMERATOR( sta );               break;
    case ST_MODULE_PROC:           print_ST_MODULE_PROC( sta );              break;
    case ST_NAMELIST:              print_ST_NAMELIST( sta );                 break;
    case ST_STATEMENT_FUNCTION:    print_ST_STATEMENT_FUNCTION( sta );       break;
    case ST_ENTRY:                 print_ST_ENTRY( sta );                    break;
    case ST_IMPORT:                print_ST_IMPORT( sta );                   break;
    default:
    break;
  }


  if( ! g95x_option.code_only ) {
    if( g95x_option.ext_locs_in_statement ) {
      g95x_print( " ext_locs=\"" );
      g95x_print_statement_ext_locs( sta );
      g95x_print( "\"" );
    } else {
      int number = sta->n_ext_locs;
      g95_file *f = sta->where.lb1->file;
      g95x_print( 
        " ext_locs_index=\"[%d,%d]\"", 
        f->x.ext_locs_index,
        number
      );
      f->x.ext_locs_index += number;

    }
  }

  print_code( sta );
  

  g95x_print( "/>\n" );
  


  if( sta->code ) {
    g95x_traverse_callback xtc;
    g95x_init_callback( &xtc );
  
    xtc.expr_callback        = _print_expr;
    xtc.constructor_callback = _print_constructor;
    xtc.ref_callback         = _print_ref;
    xtc.io_var_callback      = _print_io_var;
    
    g95x_traverse_code( sta, &xtc );
    
  }
  
  
  switch( sta->type ) {
    case ST_FORMAT:        print_statement_parts_ST_FORMAT( sta );        break;
    case ST_DATA_DECL:     print_statement_parts_ST_DATA_DECL( sta );     break;
    default:
    break;
  }
  


}

void g95x_print_string( const char *s, int l, int e ) {
  int i;
  for( i = 0; ( *s != 0 ) || ( i < l ); i++, s++ ) {
    switch( *s ) {
      case '"':
	if( e ) g95x_print( "\\" );
        g95x_print( "&quot;" );
      break;
      case '<':
        g95x_print( "&lt;" );
      break;
      case '>':
        g95x_print( "&gt;" );
      break;
      case '&':
        g95x_print( "&amp;" );
      break;
      case '\\':
        g95x_print( "\\\\" );
      break;
      default:
        g95x_print( "%c", *s );
    }
  }
}


static void print_constant( g95_expr *h, int noboz ) {

  if( ( ! noboz ) && h->x.boz ) 
    g95x_print( "%s", h->x.boz );
  
  else
 
  switch( h->ts.type ) {
    case BT_INTEGER:
      g95x_print( "%s", bi_to_string( h->value.integer ) );
    break;
    case BT_REAL:
      g95x_print( "%s", bg_to_string( h->value.real ) );
    break;
    case BT_COMPLEX:
      /* bg_to_string not re-entrant, proceed in two steps */
      g95x_print( "(%s,", bg_to_string( h->value.complex.r ) );
      g95x_print( "%s)", bg_to_string( h->value.complex.i ) );
    break;
    case BT_LOGICAL:
      g95x_print( "%s", h->value.logical ? ".TRUE." : ".FALSE." );
    break;
    case BT_CHARACTER: 
      g95x_print_string( h->value.character.string, h->value.character.length, 0 );
    break;
    case BT_DERIVED: 
    break;
    case BT_UNKNOWN: 
    break;
    case BT_PROCEDURE: 
    break;
  }    
} 


static void print_constructor( g95_constructor *c ) {    
    
  g95x_print( 
    "<constructor id=\"%R\" expr=\"%R\"", c, c->expr
  );
  
  if( c->iterator ) {
    g95x_print( " iterator=\"" );
    g95x_print( "%R", c->iterator->var );
    g95x_print( ":%R", c->iterator->start );
    g95x_print( ":%R", c->iterator->end );
    
    
    /* Avoid artificial step */
    if( g95x_nodummy_expr( c->iterator->step ) ) 
      g95x_print( ":%R", c->iterator->step );
    g95x_print( "\"" );
  }

  g95x_print( "/>\n" );
       
}       


static void print_expr_op_as_function( g95_expr *h ) {
  if( h->x.op == INTRINSIC_USER ) 
    g95x_print(  " op=\"%R\"", h->x.uop ); 
  else 
    g95x_print(  " op=\"%R\"", h->x.iop ); 
  g95_actual_arglist *a = h->value.function.actual;
  g95x_print( " op1=\"%R\"", a->u.expr );
  if( a->next )
    g95x_print( " op2=\"%R\"", a->next->u.expr );
  g95x_print( " procedure=\"%R\"", h->symbol );
}	

static void print_actual_arg_list( g95_actual_arglist *a ) {
  int printed = 0;
  g95x_print( " args=\"" );
  for( ; a; a = a->next ) {
    if( printed && a->u.expr )
      g95x_print( "," );
    if( a->name ) g95x_print( "%s=", a->name );
    if( a->type == ARG_ALT_RETURN ) {
      g95x_print( "%R", a->u.label );
      printed++;
    } else if( a->u.expr ) {
      g95x_print( "%R", a->u.expr ); 
      printed++;
    } 
  }
  g95x_print( "\"" );
}



static void print_symbol_alias( 
  g95x_locus* xw, void* generic, g95_symbol* symbol,
  const char* name
) {
  char string[G95_MAX_SYMBOL_LEN+1];
  
  if( symbol_intrinsic( symbol ) ) {
    g95_intrinsic_sym* isym = g95x_get_intrinsic_by_symbol( symbol );
    if( name )
    g95x_print( " %s=\"%R\"", name, isym );
    else
      g95x_print( "%R", isym );
    return;
  }
  
  if( ! symbol->x.use_assoc ) {
    if( name )
      g95x_print( " %s=\"%R\"", name, symbol );
    else
      g95x_print( "%R", symbol );
    return;
  }
  
  g95x_get_code_substring( string, xw, 0, G95_MAX_SYMBOL_LEN, 1 );

  if( name ) {
    if( ( ! generic ) && strcmp( symbol->name, string ) ) {
      g95x_print( " %s=\"%A\"", name, symbol, string ); 
    } else  
      g95x_print( " %s=\"%R\"", name, symbol );
  } else {
    if( ( ! generic ) && strcmp( symbol->name, string ) ) {
      g95x_print( "%A", symbol, string ); 
    } else  
      g95x_print( "%R", symbol );
  }

}


static void print_expr( g95_expr *h, int reduced ) {

       
  if( h == NULL )       
    return;    
  
  g95x_print( "<expr id=\"%R\" type=\"%s\"", h, sort_expr( h ) );
  
  if( h->type == EXPR_CONSTANT ) 
    g95x_print( " constant_type=\"%R\"", type_id( &h->ts ) );

  if( ! g95x_option.code_only ) {
    g95x_print( " f=\"%R\" loc=\"", h->x.where.lb1->file );
    print_locs( &h->x.where );
    g95x_print( "\"" ); 
  }
 

  if( reduced )
    goto done;
  

  switch( h->type ) { 
    case EXPR_SUBSTRING: 

      g95x_print( " string=\"" );
      print_constant( h, 0 );
      g95x_print( "\"" );
      g95x_print( " range=\"" );
      if( h->ref->u.ss.start ) 
        g95x_print( "%R", h->ref->u.ss.start );
      g95x_print( ":" );
      if( h->ref->u.ss.end ) 
        g95x_print( "%R", h->ref->u.ss.end );
      g95x_print( "\"" );
    break;      
      
    case EXPR_STRUCTURE: {
      g95_constructor *c;
      int printed = 0;
      g95x_print( "  args=\"" );
      for( c = h->value.constructor.c; c; c = c->next ) {
        g95_expr *e = c->expr;
        if( c->x.dummy )
          continue;
        if( printed ) 
          g95x_print( "," ); 
        if( c->x.name ) 
          g95x_print( "%s=", c->x.name );
        
        e = e->x.full ? e->x.full : e;
        g95x_print( "%R", c->expr );
        printed++;
      }
      g95x_print( "\"" );
      
      print_symbol_alias( &h->x.where, NULL, h->symbol, "symbol" );
      
    } break;      
      
    case EXPR_ARRAY: {       
      g95_constructor *c;
      g95_ref *r;
      g95x_print( " constructor=\"" );
      for( c = h->value.constructor.c; c; c = c->next ) {
        g95x_print( "%R", c );
        if( c->next ) g95x_print( "," ); 
      }
      g95x_print( "\"" );
      
      if( h->x.has_ts ) 
        print_typespec( &h->ts, "cast_" );

      if( h->ref ) {
        g95x_print( " refs=\"" );
        for( r = h->ref; r; r = r->next ) { 
          g95x_print( "%R", r );
          if( r->next ) g95x_print( "," );        
        } 
        g95x_print( "\"" );
      }

         
    } break;     
     
    case EXPR_NULL:  break;      
      
    case EXPR_CONSTANT: 
      if( h->x.boz )
        g95x_print( " boz=\"1\"" );
      if( ( h->ts.type == BT_CHARACTER ) 
        && h->value.character.hollerith )
        g95x_print( " hollerith=\"1\"" );
      g95x_print( " value=\"" );
      print_constant( h, 0 );
      g95x_print( "\"" );
      if( h->ts.x.kind ) 
        g95x_print( " kind=\"%R\"", h->ts.x.kind->kind );
    break;
     
    case EXPR_VARIABLE: {   
      g95_ref *r;
#ifdef G95X_CHECK
      if( h->symbol->attr.flavor == FL_UNKNOWN ) 
        g95x_die( "Unknown symbol flavor" );
#endif       
      
      if( h->ref ) {
        g95x_print( " refs=\"" );
      
        for( r = h->ref; r; r = r->next ) { 
          g95x_print( "%R", r );
          if( r->next ) g95x_print( "," );        
        } 
        g95x_print( "\"" );
      }
      
      print_symbol_alias( &h->x.where, NULL, h->symbol, "symbol" );
      
    
    } break;

    case EXPR_OP:    
    
      if( h->value.op.operator == INTRINSIC_USER )
        g95x_die( "Unexpected USER operator" );
      else
        g95x_print(  " op=\"%R\"", sort_iop( h->value.op.operator ) ); 
        
      if( h->symbol )
        g95x_print( " procedure=\"%R\"", h->symbol );
    
      g95x_print( " op1=\"%R\"", h->value.op.op1 ); 
    
      if( h->value.op.op2 )     
        g95x_print( " op2=\"%R\"", h->value.op.op2 );         
    
        
    break;

    case EXPR_FUNCTION: {
      
#ifdef G95X_CHECK
        if( ! h ) 
          g95x_die( "Function expr symbol is null\n" );
#endif
    
      /* user/intrinsic extended operator */
      if( h->x.is_operator ) 
        print_expr_op_as_function( h );
      else {
    
        print_actual_arg_list( h->value.function.actual );
        
        if( h->x.generic )  /* this is for resolved generic function calls */
          g95x_print( " generic=\"%R\"", h->x.generic );
    
        if( h->symbol ) { /* ordinary function call */
          print_symbol_alias( 
	    &h->x.where, h->x.generic, 
	    h->symbol, "function" 
	  );
	} else if( h->value.function.pointer ) 
	  g95x_print( " function=\"%R\"", h->value.function.pointer );
        else 
          g95x_print( " function=\"%R\"", g95x_get_intrinsic_by_symbol( h->x.symbol ) );
      }
         
    } break;    
    
    case EXPR_UNKNOWN:          
    break;   
   
    case EXPR_PROCEDURE:    
    
      print_symbol_alias( &h->x.where, NULL, h->symbol, "symbol" );
          
    break;   
   
  }  

done:  
  g95x_print( "/>\n" );
  return;

}    

static void print_code( g95x_statement *sta ) {
  g95_forall_iterator *fa;     
  g95_open *open;      
  g95_alloc *b;  
  g95_close *close;    
  g95_filepos *f;      
  g95_inquire *s;    
  g95_dt *dt;    
  g95_code *p, code;
  

  if( ! sta->code )
    return;


  code = *( sta->code );
  p = &code;
    
  if( ( sta->type == ST_ASSIGNMENT ) 
   && ( p->type == EXEC_CALL ) ) {
    /* This means that our assignment has been replaced
       by a call; hence it is an overloaded assignment.
       so we fake an EXEC_ASSIGN code */
    p->type = EXEC_ASSIGN;
    p->expr = p->ext.sub.actual->u.expr;
    p->expr2 = p->ext.sub.actual->next->u.expr;
  }

  if( sta->type == ST_ARITHMETIC_IF ) {
    if( p->label )
      g95x_print( " label1=\"%R\"", p->label );
    if( p->label2 )
      g95x_print( " label2=\"%R\"", p->label2 );
    if( p->label3 )
      g95x_print( " label3=\"%R\"", p->label3 );
  }

  switch( sta->type ) {
    case ST_SIMPLE_IF: case ST_WHERE:
      if( p->expr )
        g95x_print( " expr=\"%R\"", p->expr );
      else if( p->block && p->block->expr ) 
        g95x_print( " expr=\"%R\"", p->block->expr );
      return;
    break;
    case ST_NULLIFY:
      p->expr2 = NULL;
    break;
    default:
    break;
  }
  
  
  /* computed goto */
  if( ( sta->type == ST_GOTO ) && ( p->type == EXEC_SELECT ) ) {
    g95_code *q;
    g95x_print( " expr=\"%R\"", p->expr2 );
    g95x_print( " goto_labels=\"" );
    for( q = p->block; q; q = q->block ) {
      g95x_print( "%R", q->next->label );
      if( q->block )
        g95x_print( "," );
    }
    g95x_print( "\"" );
    return;
  }
 
  switch( p->type ) { 
    case EXEC_NOP: break;     
       
    case EXEC_CONTINUE: break;    
      
    case EXEC_AC_ASSIGN:
      g95x_print( " expr=\"%R\"", p->expr );         
    break;       
         
    case EXEC_ASSIGN:
    case EXEC_POINTER_ASSIGN:
    
    if( sta->type == ST_NULLIFY ) {
      g95_code* c;
      int n = 1;
      g95x_print( " args=\"" );
      for( c = p; ; c = c->next, n++ ) {
        g95x_print( "%R", c->expr );
        if( n == sta->u.nullify.n )
          break;
        if( c->next ) 
          g95x_print( "," );
      }
      g95x_print( "\"" );
      
    } else {
      if( p->expr ) 
        g95x_print( " expr1=\"%R\"", p->expr ); 
      if( p->expr2 ) 
        g95x_print( " expr2=\"%R\"", p->expr2 );
      if( p->sym ) /* this is an overloaded assignment */
        g95x_print( " procedure=\"%R\"", p->sym );
    } 
    break; 
   
    case EXEC_GOTO: 
      if( p->label ) 
        g95x_print( " goto_label=\"%R\"", p->label );
      else if( p->expr )
        g95x_print( " expr=\"%R\"", p->expr );
    
    break;       
         
    case EXEC_CALL: {
      {
        g95_symbol *sym = p->sym ? p->sym : p->x.symbol;
        if( sym->attr.flavor == FL_UNKNOWN ) 
          g95x_die( "Unknown symbol flavor" );
      }

      print_actual_arg_list( p->ext.sub.actual );

      if( sta->type == ST_CALL ) {
        if( p->ext.sub.pointer ) 
	  g95x_print( " subroutine=\"%R\"", p->ext.sub.pointer );
	else {

          if( p->sym ) { /* ordinary call */
            g95_symbol *symbol =  p->sym;
            g95x_ext_locus *xw = NULL;
            int i;
            for( i = 0; i < sta->n_ext_locs; i++ ) 
              if( ( sta->ext_locs[i].type == G95X_LOCUS_SMBL )
               || ( sta->ext_locs[i].type == G95X_LOCUS_GNRC )
              ) {
                xw = &sta->ext_locs[i];
                goto found;
              }
            g95x_die( "Cannot find name locus in ST_CALL\n" );
found:        
            if( p->x.generic ) 
              g95x_print( " generic=\"%R\"", p->x.generic );

            print_symbol_alias( &xw->loc, p->x.generic, symbol, "subroutine" );

          } else 
            g95x_print( " subroutine=\"%R\"", g95x_get_intrinsic_by_symbol( p->x.symbol ) );
	}
      }
      break;
    }
    case EXEC_RETURN:        
      if( p->expr ) 
        g95x_print( " expr=\"%R\"", p->expr );         
    break;     
       
    case EXEC_STOP:          
    case EXEC_ALL_STOP:
    case EXEC_PAUSE:       
        
      if( p->expr )      
        g95x_print( " expr=\"%R\"", p->expr );         
      else if( p->ext.stop_code > 0 )
        g95x_print( " value=\"%d\"", p->ext.stop_code );
   
    break;         
           
    case EXEC_ARITHMETIC_IF:   
            
      g95x_print( " expr=\"%R\"", p->expr );         
       
    break;       
         
    case EXEC_IF:       
      g95x_print( " expr=\"%R\"", p->expr );  
            
    break;     
       
    case EXEC_SELECT:   
    
      if( p->expr ) 
        g95x_print( " expr=\"%R\"", p->expr );
      else if( p->expr2 )
        g95x_print( " expr=\"%R\"", p->expr2 );
      else {
        g95_expr *lo = p->ext.case_list->low;
        g95_expr *hi = p->ext.case_list->high;
        if( lo && lo->x.full ) lo = lo->x.full;
        if( hi && hi->x.full ) hi = hi->x.full;
        if( lo || hi ) {
          g95x_print( " case=\"" );
          if( lo )
            g95x_print( "%R", lo );
          if( lo != hi )
            g95x_print( ":" );
          if( hi && ( lo != hi ) )
            g95x_print( "%R", hi );
          g95x_print( "\"" );
        }
      }
      
   
    break;   
     
    case EXEC_WHERE:       
      if( p->block )
        g95x_print( " expr=\"%R\"", p->block->expr );         
   
    break;
   
    case EXEC_FORALL:
      if( p->ext.forall_iterator ) {
        g95x_print( " iterators=\"" );
        for( fa = p->ext.forall_iterator; fa; fa = fa->next ) {     
   
          g95x_print( "%R", fa->var );   
          g95x_print( ":%R", fa->start );         
          g95x_print( ":%R", fa->end );         
          if( g95x_nodummy_expr( fa->stride ) )    
            g95x_print( ":%R", fa->stride );         
         
          if( fa->next ) 
            g95x_print( "," );   
        } 
        g95x_print( "\"" );
      }       
          
      if( p->expr ) {     
        g95x_print( " expr=\"%R\"", p->expr );         
      }       
   
    break;    
          
    case EXEC_DO: 
      if( p->label ) 
        g95x_print( " do_label=\"%R\"", p->label );
      g95x_print( " iterator=\"" );
      g95x_print( "%R", p->ext.iterator->var );   
      g95x_print( ":%R", p->ext.iterator->start ); 	
      g95x_print( ":%R", p->ext.iterator->end );
      if( g95x_nodummy_expr( p->ext.iterator->step ) )    
        g95x_print( ":%R", p->ext.iterator->step );	 
      g95x_print( "\"" );
   
    break;  
    
    case EXEC_DO_WHILE:          
      /* Changed transform_while */
      if( p->label ) 
        g95x_print( " do_label=\"%R\"", p->label );
      if( p->expr )
        g95x_print( " while=\"%R\"", p->expr );
   
    break;  
    
    case EXEC_EXIT:
    case EXEC_CYCLE:
      if( p->sym )
        g95x_print( " block=\"%R\"", p->sym );
    break;   
     
    case EXEC_DEALLOCATE:      
    case EXEC_ALLOCATE:
      
      if( p->expr )
        g95x_print( " stat=\"%R\"", p->expr );

      g95x_print( " exprs=\"" );
      for( b = p->ext.alloc_list; b; b = b->next ) {        
        g95x_print( "%R", b->expr );         
        if( b->next != NULL )
          g95x_print( "," );    
      }          
      g95x_print( "\"" );
            
    break; 
   
    case EXEC_OPEN:
      open = p->ext.open;    
            
      if( open->unit )  
        g95x_print( " unit=\"%R\"", open->unit );   
          
      if( open->iostat )    
        g95x_print( " iostat=\"%R\"", open->iostat );   
          
      if( open->file )       
        g95x_print( " file=\"%R\"", open->file );   
        
      if( open->status )    
        g95x_print( " status=\"%R\"", open->status );   
    
      if( open->access )     
        g95x_print( " access=\"%R\"", open->access );   
          
      if( open->form )          
        g95x_print( " form=\"%R\"", open->form );   
   
      if( open->recl )   
        g95x_print( " recl=\"%R\"", open->recl );   
   
      if( open->blank )  
        g95x_print( " blank=\"%R\"", open->blank );   
         
      if( open->position )  
        g95x_print( " position=\"%R\"", open->position );   
       
      if( open->action )       
        g95x_print( " action=\"%R\"", open->action );   
         
      if( open->delim )   
        g95x_print( " delim=\"%R\"", open->delim );   
            
      if( open->pad )    
        g95x_print( " pad=\"%R\"", open->pad );   
   
      if( open->decimal )    
        g95x_print( " pad=\"%R\"", open->decimal );   
   
      if( open->err )     
        g95x_print( " err=\"%R\"", open->err );   
      
    break;        
          
    case EXEC_CLOSE:  
      close = p->ext.close;  
         
      if( close->unit )  
        g95x_print( " unit=\"%R\"", close->unit );   
   
      if( close->iostat )   
        g95x_print( " iostat=\"%R\"", close->iostat );   
   
      if( close->status )    
        g95x_print( " status=\"%R\"", close->status );   
         
      if( close->err != NULL )           
        g95x_print( " err=\"%R\"", close->err );   
       
    break;        
          
    case EXEC_BACKSPACE:       
    case EXEC_ENDFILE:  
    case EXEC_REWIND:     
     
      f = p->ext.filepos;    
            
      if( f->unit )   
        g95x_print( " unit=\"%R\"", f->unit );   
    
      if( f->iostat )       
        g95x_print( " iostat=\"%R\"", f->iostat );   
    
      if( f->err )      
        g95x_print( " err=\"%R\"", f->err );   
        
    break;          
            
    case EXEC_INQUIRE:  
      s = p->ext.inquire;      
  
    
    
    
      if( s->unit )           
        g95x_print( " unit=\"%R\"", s->unit );   
         
      if( s->file )         
        g95x_print( " file=\"%R\"", s->file );   
    
      if( s->iostat )      
        g95x_print( " iostat=\"%R\"", s->iostat );   
            
      if( s->exist )    
        g95x_print( " exist=\"%R\"", s->exist );   
       
      if( s->opened )       
        g95x_print( " opened=\"%R\"", s->opened );   
     
      if( s->number )     
        g95x_print( " number=\"%R\"", s->number );   
   
      if( s->named )    
        g95x_print( " named=\"%R\"", s->named );   
          
      if( s->name )  
        g95x_print( " name=\"%R\"", s->name );   
     
      if( s->access ) 	    
        g95x_print( " access=\"%R\"", s->access );   
          
      if( s->sequential )  
        g95x_print( " sequential=\"%R\"", s->sequential );   
           
      if( s->direct ) 	    
        g95x_print( " direct=\"%R\"", s->direct );   
          
      if( s->form )   
        g95x_print( " form=\"%R\"", s->form );   
        
      if( s->formatted )    
        g95x_print( " formatted=\"%R\"", s->formatted );   
     
      if( s->unformatted )     
        g95x_print( " unformatted=\"%R\"", s->unformatted );   
   
      if( s->recl ) 	 
        g95x_print( " recl=\"%R\"", s->recl );   
           
      if( s->nextrec )  
        g95x_print( " nextrec=\"%R\"", s->nextrec );   
          
      if( s->blank )     
        g95x_print( " blank=\"%R\"", s->blank );   
        
      if( s->position ) 	      
        g95x_print( " position=\"%R\"", s->position );   
        
      if( s->action )     
        g95x_print( " action=\"%R\"", s->action );   
        
      if( s->read )      
        g95x_print( " read=\"%R\"", s->read );   
         
      if( s->write ) 	    
        g95x_print( " write=\"%R\"", s->write );   
         
      if( s->readwrite )     
        g95x_print( " readwrite=\"%R\"", s->readwrite );   
        
      if( s->delim ) 	   
        g95x_print( " delim=\"%R\"", s->delim );   
     
      if( s->pad )    
        g95x_print( " pad=\"%R\"", s->pad );   
         
      if( s->pos )    
        g95x_print( " pos=\"%R\"", s->pos );   
         
      if( s->iolength )    
        g95x_print( " iolength=\"%R\"", s->iolength );   
         
      if( s->size )    
        g95x_print( " size=\"%R\"", s->size );   
         
      if( s->err != NULL ) 
        g95x_print( " err=\"%R\"", s->err );	 

   
    break;         

    case EXEC_IOLENGTH:  
      g95x_print( " iolength=\"%R\"", p->expr );    
      goto print_transfers;   
        
    case EXEC_READ:       
    case EXEC_WRITE: {
      g95_code *c;   
      int fmt = 0;
      dt = p->ext.dt;     
     
      if( g95x_nodummy_expr( dt->io_unit ) ) 
        g95x_print( " unit=\"%R\"", dt->io_unit ); 
      else
        g95x_print( " unit=\"*\"" ); 
          
      if( g95x_nodummy_expr( dt->format_expr ) && ( dt->format_label != NULL ) )
        if( dt->format_label->value > 0 ) 
          g95x_die( "Strange io statement" );

      if( g95x_nodummy_expr( dt->format_expr ) ) {	  
        g95x_print( " fmt=\"%R\"", dt->format_expr );
        fmt = 1;   
      } else {
        if( dt->format_label != NULL ) 
          if( dt->format_label->value > 0 ) {
            g95x_print( " fmt=\"%R\"", dt->format_label );   
            fmt = 1;
          }
      }
      if( ( fmt == 0 ) 
       && ( ! dt->namelist ) 
       && ( ! g95x_nodummy_expr( dt->rec ) ) 
      ) 
        g95x_print( " fmt=\"*\"" );
        
      if( dt->namelist ) 
        g95x_print( " nml=\"%R\"", dt->namelist );   
        
      if( g95x_nodummy_expr( dt->iostat ) ) 
        g95x_print( " iostat=\"%R\"", dt->iostat );   
          
      if( g95x_nodummy_expr( dt->size ) ) 
        g95x_print( " size=\"%R\"", dt->size );   
        
      if( g95x_nodummy_expr( dt->rec ) ) 
        g95x_print( " rec=\"%R\"", dt->rec );   
      
      if( g95x_nodummy_expr( dt->advance ) )  
        g95x_print( " advance=\"%R\"", dt->advance );   
   
      if( g95x_nodummy_expr( dt->pos ) ) 
        g95x_print( " pos=\"%R\"", dt->pos );   
   
      if( g95x_nodummy_expr( dt->decimal ) )  
        g95x_print( " decimal=\"%R\"", dt->decimal );   
   
print_transfers: {
        int end = 0;
        g95x_print( " args=\"" );
        for( c = p->next; c; c = c->next ) {
          switch( c->type ) {
            case EXEC_TRANSFER:
              g95x_print( "%R", c->expr );
              if( c->next->type != EXEC_DT_END ) 
                g95x_print( "," );
            break;
            case EXEC_DO:
              g95x_print( "%R", c->block );
              if( c->next->type != EXEC_DT_END ) 
                g95x_print( "," );
            break;
	    default:
	    break;
          }
          if( c->type == EXEC_DT_END ) {
            g95x_print( "\"" );
            end++;
            dt = c->ext.dt;
            if( dt != NULL ) {  
              if( dt->err ) 
                g95x_print( " err=\"%R\"", dt->err ); 
              if( dt->end )        
                g95x_print( " end=\"%R\"", dt->end ); 
              if( dt->eor )   
                g95x_print( " eor=\"%R\"", dt->eor ); 
            }          
          
            break;
          } 
        }
        if( ! end ) 
          g95x_print( "\"" );
      }
    } 
    
    break;

    case EXEC_FLUSH: {
      g95_flush* f = p->ext.flush;
      if( g95x_nodummy_expr( f->unit ) ) 
        g95x_print( " unit=\"%R\"", f->unit );    
      if( g95x_nodummy_expr( f->iostat ) ) 
        g95x_print( " iostat=\"%R\"", f->iostat );   
      if( g95x_nodummy_expr( f->iomsg ) ) 
        g95x_print( " iomsg=\"%R\"", f->iomsg );   
      if( f->err ) 
        g95x_print( " err=\"%R\"", f->err ); 
    }
    break;

    case EXEC_WAIT: {
      g95_wait* f = p->ext.wait;
      if( g95x_nodummy_expr( f->unit ) ) 
        g95x_print( " unit=\"%R\"", f->unit );    
      if( g95x_nodummy_expr( f->id ) ) 
        g95x_print( " id=\"%R\"", f->id );   
      if( g95x_nodummy_expr( f->iostat ) ) 
        g95x_print( " iostat=\"%R\"", f->iostat );   
      if( g95x_nodummy_expr( f->iomsg ) ) 
        g95x_print( " iomsg=\"%R\"", f->iomsg );   
      if( f->err ) 
        g95x_print( " err=\"%R\"", f->err ); 
      if( f->end ) 
        g95x_print( " end=\"%R\"", f->end ); 
      if( f->eor ) 
        g95x_print( " eor=\"%R\"", f->eor ); 
    }
    break;

    case EXEC_TRANSFER:      
    case EXEC_DT_END: 
      g95x_die( "Attempt to print TRANSFER/DT_END code\n" );    
    break;   
     
    case EXEC_ENTRY:        
    break; 
   
    case EXEC_LABEL_ASSIGN:
      g95x_print( " assign_label=\"%R\" symbol=\"%R\"", p->label, p->sym );    
    break;     
       
    case EXEC_SYNC_ALL:
    case EXEC_SYNC_TEAM:
    case EXEC_SYNC_IMAGES:
    case EXEC_SYNC_MEMORY:
    
      if( g95x_nodummy_expr( p->expr ) ) 
        g95x_print( " expr=\"%R\"", p->expr );   
      
      if( g95x_nodummy_expr( p->ext.sync.stat ) ) 
        g95x_print( " stat=\"%R\"", p->ext.sync.stat );   
      
      if( g95x_nodummy_expr( p->ext.sync.errmsg ) ) 
        g95x_print( " errmsg=\"%R\"", p->ext.sync.errmsg );   
      
      
    break;
    
    case EXEC_CRITICAL:
    break;
    
    


    case EXEC_AC_START:
    case EXEC_WHERE_ASSIGN:
    
    default:
      g95x_die( "Unknown code\n" );

  }          

}


/*
 * Here, we dump all intrinsics in a module INTRINSIC
 * we have to retrieve all info from intrinsic.c
 */

static void print_intrinsic_args( g95_intrinsic_sym *s ) {
  char buffer[256];
  g95_intrinsic_arg *a;

  for( a = s->formal; a; a = a->next ) { 
  
    sprintf( buffer, REFFMT1, a );
    g95x_print( "<symbol id=\"%A\" name=\"%s\" flavor=\"VARIABLE\" dummy=\"1\"", s, buffer, a->name );
    switch( a->intent ) {
      case INTENT_IN:    g95x_print( " intent=\"IN\"" );    break;
      case INTENT_OUT:   g95x_print( " intent=\"OUT\"" );   break;
      case INTENT_INOUT: g95x_print( " intent=\"INOUT\"" ); break;
      default:
      break;
    }
    if( a->optional ) 
      g95x_print( " optional=\"1\"" );
    print_typespec( &a->ts, "" );
    g95x_print( "/>\n" );
  
  }
}

static void print_intrinsic_arg_list( g95_intrinsic_sym *s ) {
  char buffer[256];
  g95_intrinsic_arg *a;

  g95x_print( " symbols=\"" );
  for( a = s->formal; a; a = a->next ) { 
  
    sprintf( buffer, REFFMT1, a );
    g95x_print( "%A", s, buffer );
    
    if( a->next )
      g95x_print( "," );
    
  }
  g95x_print( "\"" );
}

static void print_intrinsic1( g95_intrinsic_sym *s, const char* flavor ) {
  g95_intrinsic_arg *a;
  g95_intrinsic_sym *t;

  g95x_print( "<symbol id=\"%R\"", s );
  
  if( strcmp( s->name, "" ) ) 
    g95x_print( " name=\"%s\" access=\"PUBLIC\"", s->name );
  else
    g95x_print( " name=\"I_%R\" access=\"PRIVATE\"", s );

  g95x_print( " flavor=\"%s\"", flavor );
  
  print_typespec( &s->ts, "" );
  
  if( s->elemental )
    g95x_print( " elemental=\"1\"" );
  if( s->pure )
    g95x_print( " pure=\"1\"" );
  if( s->generic )
    g95x_print( " generic=\"1\"" );
  
  if( s->formal ) {
    char buffer[256];
    g95x_print( " args=\"" );
    for( a = s->formal; a; a = a->next ) {
      sprintf( buffer, REFFMT1, a );
      g95x_print( "%A", s, buffer );
      if( a->next )
        g95x_print( "," );
    }
    g95x_print( "\"" );
  }
  
  if( s->specific_head ) {
    g95x_print( " specifics=\"" );
    for( t = s->specific_head; t; t = t->next ) {
      g95x_print( "%R", t );
      if( t->next )
        g95x_print( "," );
    }
    g95x_print( "\"" );
  }
  
  g95x_print( "/>\n" );
  
  
  
}

static void print_intrinsic2( g95_intrinsic_sym *s ) {

  g95x_print( "<namespace id=\"%R\" symbol=\"%R\"", (char*)s + 1, s );
  
  print_intrinsic_arg_list( s );
  
  g95x_print( "/>\n" );

  print_intrinsic_args( s );
}

static void print_intrinsic_generic( g95_intrinsic_sym *s ) {
g95_intrinsic_sym *t;

  if( ! s->specific_head )
    return;
    
  g95x_print( 
    "<generic id=\"%R\" name=\"%s\" procedures=\"%R,", 
    (char*)s + 2, s->name, s 
  );
  for( t = s->specific_head; t; t = t->next ) {
    g95x_print( "%R", t );
    if( t->next )
      g95x_print( "," );
  }
  g95x_print( "\"/>\n" );
}

void print_intrinsics_interface_ns_list() {
  int i;
  g95_intrinsic_sym *f, *s, *is;
  int nf, ns;
  int printed = 0;

  g95x_get_intrinsics_info( &f, &s, &nf, &ns );
  
  g95x_print( " interface_namespaces=\"" );
  for( i = 0; i < nf; i++ ) {
    is = f + i;
    if( printed )
      g95x_print( "," );
    g95x_print( "%R", (char*)is + 1 );
    printed++;
  }
  for( i = 0; i < ns; i++ ) {
    is = s + i;
    if( printed )
      g95x_print( "," );
    g95x_print( "%R", (char*)is + 1 );
    printed++;
  }
  g95x_print( "\"" );
}

void print_intrinsics_symbol_list() {
  int i;
  g95_intrinsic_sym *f, *s, *is;
  int nf, ns;

  g95x_get_intrinsics_info( &f, &s, &nf, &ns );
  
  g95x_print( " symbols=\"%R", (char*)g95x_print_intrinsics + 1 );
  
  for( i = 0; i < nf; i++ ) {
    is = f + i;
    g95x_print( ",%R", is );
  }
  for( i = 0; i < ns; i++ ) {
    is = s + i;
    g95x_print( ",%R", is );
  }

  g95x_print( "\"" );
}

void print_intrinsics_symbols() {
  int i;
  g95_intrinsic_sym *f, *s, *is;
  int nf, ns;

  g95x_get_intrinsics_info( &f, &s, &nf, &ns );

  g95x_print( "<symbol id=\"%R\" name=\"INTRINSIC\" flavor=\"MODULE\"/>\n", 
  (char*)g95x_print_intrinsics + 1 );
  
  for( i = 0; i < nf; i++ ) {
    is = f + i;
    print_intrinsic1( is, "FUNCTION" );
  }
  for( i = 0; i < ns; i++ ) {
    is = s + i;
    print_intrinsic1( is, "SUBROUTINE" );
  }

}

void print_intrinsics_generic_list() {
  int i;
  g95_intrinsic_sym *f, *s, *is;
  int nf, ns;
  int printed = 0;

  g95x_get_intrinsics_info( &f, &s, &nf, &ns );

  g95x_print( " generics=\"" );
  for( i = 0; i < nf; i++ ) {
    is = f + i;
    if( ! is->specific_head )
      continue;
    if( printed )
      g95x_print( "," );
    g95x_print( "%R", (char*)is + 2 );
    printed++;
  }
  for( i = 0; i < ns; i++ ) {
    is = s + i;
    if( ! is->specific_head )
      continue;
    if( printed )
      g95x_print( "," );
    g95x_print( "%R", (char*)is + 2 );
    printed++;
  }
  g95x_print( "\"" );


}


void print_intrinsics_generics() {
  int i;
  g95_intrinsic_sym *f, *s, *is;
  int nf, ns;

  g95x_get_intrinsics_info( &f, &s, &nf, &ns );
  for( i = 0; i < nf; i++ ) {
    is = f + i;
    print_intrinsic_generic( is );
  }
  for( i = 0; i < ns; i++ ) {
    is = s + i;
    print_intrinsic_generic( is );
  }
}

void g95x_print_intrinsics() {
  int i;
  g95_intrinsic_sym *f, *s, *is;
  int nf, ns;

  g95x_get_intrinsics_info( &f, &s, &nf, &ns );

  
  g95x_print( 
    "<namespace id=\"%R\" symbol=\"%R\"", 
    g95x_print_intrinsics, (char*)g95x_print_intrinsics + 1
  );
  
  print_intrinsics_interface_ns_list();
  
  print_intrinsics_symbol_list();
  
  print_intrinsics_generic_list();
  
  g95x_print( "/>\n" );
  
  print_intrinsics_symbols();
  
  print_intrinsics_generics();
  

  
  for( i = 0; i < nf; i++ ) {
    is = f + i;
    print_intrinsic2( is );
  }
  for( i = 0; i < ns; i++ ) {
    is = s + i;
    print_intrinsic2( is );
  }
  
}


static void print_iops() {
  const char *fmt = "<operator id=\"%R\" name=\"%s\" type=\"INTRINSIC\"/>\n";
  const char *p;
  
  p = sort_iop( INTRINSIC_UPLUS );   g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_UMINUS );  g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_PLUS );    g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_MINUS );   g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_TIMES );   g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_DIVIDE );  g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_POWER );   g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_CONCAT );  g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_AND );     g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_OR );      g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_EQV );     g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_NEQV );    g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_EQ );      g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_NE );      g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_GT );      g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_GE );      g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_LT );      g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_LE );      g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_NOT );     g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_PAREN );   g95x_print( fmt, p, p );
  p = sort_iop( INTRINSIC_ASSIGN );  g95x_print( fmt, p, p );

}

static void print_ops_list() {
  const char *p;
  
  p = sort_iop( INTRINSIC_UPLUS );   g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_UMINUS );  g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_PLUS );    g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_MINUS );   g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_TIMES );   g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_DIVIDE );  g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_POWER );   g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_CONCAT );  g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_AND );     g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_OR );      g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_EQV );     g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_NEQV );    g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_EQ );      g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_NE );      g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_GT );      g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_GE );      g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_LT );      g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_LE );      g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_NOT );     g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_PAREN );   g95x_print( "%R,", p );
  p = sort_iop( INTRINSIC_ASSIGN );  g95x_print( "%R", p );

}



/* user ops base */

static void print_uops() {
  _symtree_data *sud;
  for( sud = symtree_data_uop_head; sud; sud = sud->next ) 
    g95x_print( 
      "<operator id=\"%R\" name=\"%s\" type=\"USER\"/>\n",
      sud->name, sud->name
    );
}

/* generics base */



static void print_gens() {
  _symtree_data *sud;
  g95_intrinsic_sym *isymf;
  g95_intrinsic_sym *isyms;
  
  for( sud = symtree_data_gen_head; sud; sud = sud->next ) {
    isymf = g95_find_function( sud->name );
    isyms = g95_find_subroutine( sud->name );
    
    g95x_print( 
      "<generic id=\"%R\" name=\"%s\" procedures=\"",
      sud->name, sud->name
    );
    
    if( isymf )
      g95x_print( "%R", isymf );
    if( isymf && isyms )
      g95x_print( "," );
    if( isyms )
      g95x_print( "%R", isyms );
      
    
    g95x_print( "\" type=\"%s\"", isymf || isyms ? "INTRINSIC" : "USER" );

    g95x_print( "/>\n" );
    
    if( isymf ) 
      g95x_print( "<symbol id=\"%R\" name=\"%s\" flavor=\"INTRINSIC\"/>\n", isymf, sud->name );
    
    if( isyms ) 
      g95x_print( "<symbol id=\"%R\" name=\"%s\" flavor=\"INTRINSIC\"/>\n", isyms, sud->name );
    
  }
  
    
}


/* types */

static void print_types() { /* print builtin types */
  g95_typespec ts;
  g95_integer_info *ip;
  g95_ff *fp;
  g95_logical_info *lp;
  
  int dik = g95_default_integer_kind( 1 );
  int drk = g95_default_real_kind( 1 );
  int ddk = g95_default_double_kind();
  int dlk = g95_default_logical_kind();
  
  
  ts.type = BT_INTEGER;
  for( ip = g95_integer_kinds; ip->kind != 0; ip++ ) {
    ts.kind = ip->kind;
    g95x_print( 
      "<type id=\"%R\" name=\"INTEGER\" kind=\"%d\"%s type=\"INTRINSIC\"/>\n",
      type_id( &ts ), ip->kind,
      dik == ts.kind ? " default=\"INTEGER\"" : ""
    );
  }

  ts.type = BT_REAL;
  for( fp = g95_real_kinds; fp->kind != 0; fp++ ) {
    int done = 0;
    const char *d1, *d2;
    ts.kind = fp->kind;
    d1 = ( drk == ts.kind ) && ( ++done ) ? " default=\"REAL\"" : "";
    d2 = ( done == 0 ) && ( ddk == ts.kind ) ? " default=\"DOUBLEPRECISION\"" : "";
    g95x_print( 
      "<type id=\"%R\" name=\"REAL\" kind=\"%d\"%s%s type=\"INTRINSIC\"/>\n",
      type_id( &ts ), fp->kind, d1, d2
    );
  }

  ts.type = BT_COMPLEX;
  for( fp = g95_real_kinds; fp->kind != 0; fp++ ) {
    int done = 0;
    const char *d1, *d2;
    ts.kind = fp->kind;
    d1 = ( drk == ts.kind ) && ( ++done ) ? " default=\"COMPLEX\"" : "";
    d2 = ( done == 0 ) && ( ddk == ts.kind ) ? " default=\"DOUBLECOMPLEX\"" : "";
    g95x_print( 
      "<type id=\"%R\" name=\"COMPLEX\" kind=\"%d\"%s%s type=\"INTRINSIC\"/>\n",
      type_id( &ts ), fp->kind, d1, d2
    );
  }

  ts.type = BT_LOGICAL;
  for( lp = g95_logical_kinds; lp->kind != 0; lp++ ) {
    ts.kind = lp->kind;
    g95x_print( 
      "<type id=\"%R\" name=\"LOGICAL\" kind=\"%d\"%s type=\"INTRINSIC\"/>\n",
      type_id( &ts ), lp->kind,
      dlk == ts.kind ? " default=\"LOGICAL\"" : ""
    );
  }
  
  ts.type = BT_CHARACTER;
  g95x_print( 
    "<type id=\"%R\" name=\"CHARACTER\" kind=\"1\" default=\"CHARACTER\" type=\"INTRINSIC\"/>\n",
    type_id( &ts )
  );

  ts.type = BT_PROCEDURE;
  g95x_print( 
    "<type id=\"%R\" name=\"PROCEDURE\" type=\"INTRINSIC\"/>\n",
    type_id( &ts )
  );


}

static void print_types_list() { /* print builtin types */
  g95_typespec ts;
  g95_integer_info *ip;
  g95_ff *fp;
  g95_logical_info *lp;
  
  
  
  ts.type = BT_INTEGER;
  for( ip = g95_integer_kinds; ip->kind != 0; ip++ ) {
    ts.kind = ip->kind;
    g95x_print( "%R,", type_id( &ts ) );
  }

  ts.type = BT_REAL;
  for( fp = g95_real_kinds; fp->kind != 0; fp++ ) {
    ts.kind = fp->kind;
    g95x_print( "%R,", type_id( &ts ) );
  }

  ts.type = BT_COMPLEX;
  for( fp = g95_real_kinds; fp->kind != 0; fp++ ) {
    ts.kind = fp->kind;
    g95x_print( "%R,", type_id( &ts ) );
  }

  ts.type = BT_LOGICAL;
  for( lp = g95_logical_kinds; lp->kind != 0; lp++ ) {
    ts.kind = lp->kind;
    g95x_print( "%R,", type_id( &ts ) );
  }
  
  ts.type = BT_CHARACTER;
  g95x_print( "%R,", type_id( &ts ) );

  ts.type = BT_PROCEDURE;
  g95x_print( "%R", type_id( &ts ) );


}

int g95x_start( g95_namespace *ns ) {
  int i;


  if( g95x_option.g95x_out ) {
  } else if( g95x_option.file ) {
    g95x_option.g95x_out = fopen( g95x_option.file, "w" );
  } else if( g95_source_file ) {
    g95x_option.file = g95x_f2xml( g95_source_file );
    g95x_option.g95x_out = fopen( g95x_option.file, "w" );
  } else if( g95x_option.intrinsics ) {
    g95x_option.file = g95x_f2xml( "INTRINSIC.F90" );
    g95x_option.g95x_out = fopen( g95x_option.file, "w" );
  }


  g95x_print( "<fortran95 options=\"" );
  for( i = 0; i < g95x_option.argc - 1; i++ ) {
    if( g95x_parse_arg( g95x_option.argc - i, g95x_option.argv + i, 0 ) > 0 ) 
      continue; 
    g95x_print( "%s", g95x_option.argv[i] );
    if( i < g95x_option.argc - 2 ) 
      g95x_print( "," );
  }
  
  g95x_print( "\" types=\"" );
  print_types_list();
  g95x_print( "\"" );
  
  g95x_print( " operators=\"" );
  print_ops_list();
  g95x_print( "\"" );
  
  if( ( ! g95x_option.intrinsics ) && ( ! g95x_option.code_only ) )
    g95x_print( " file_head=\"%R\"", g95x_get_file_head() );
  
  if( g95x_statement_head ) 
    g95x_print( " statement_head=\"%R\" statement_tail=\"%R\"", 
                g95x_statement_head, g95x_statement_tail );

  g95x_print( ">\n" );

  print_types();
  
  print_iops();
  
  if( ns ) {
    def_symtree( ns, 0, &symtree_data_uop_head );
    def_symtree( ns, 1, &symtree_data_gen_head );
    print_uops();
    print_gens();
  }
  	      
  return 1;
}

int g95x_end() {
  _symtree_data *sud, *sud1;
  
  for( sud = symtree_data_uop_head; sud; ) {
    sud1 = sud->next;
    g95_free( sud->name );
    g95_free( sud );
    sud = sud1;
  }
  
  for( sud = symtree_data_gen_head; sud; ) {
    sud1 = sud->next;
    g95_free( sud->name );
    g95_free( sud );
    sud = sud1;
  }
  
  if( g95x_option.check )
    g95x_check_bbt_ids();
  
  g95x_free_bbt_ids( &root_bbt_id );
  
  g95x_print( "</fortran95>\n" );
  return 1;
}

