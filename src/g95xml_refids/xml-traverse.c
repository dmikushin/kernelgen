#include "g95.h"
#include <stdio.h>
#include <stdlib.h>


/* 
 * Symtree traversal, with callback and data.
 */

void g95x_traverse_symtree( 
  g95_symtree *st, void ( *func )( g95_symtree*, void* ), void* data 
) {

  if( st != NULL ) {

    g95x_traverse_symtree( st->left, func, data );
    ( *func )( st, data );
    g95x_traverse_symtree( st->right, func, data );
    
  }
}

/* 
 * ref traversal
 */

static void traverse_ref( g95_ref* d, g95x_traverse_callback *xtc ) { 

  if( xtc->ref_callback ) {
    if( xtc->ref_callback( d, xtc->ref_data ) < 0 ) 
      return;
  }
    
 
  switch( d->type ) {  
    case REF_ARRAY: {
      g95_array_ref *a = &d->u.ar;
      int t;
      switch( a->type ) { 
        case AR_FULL: break; 
        case AR_SECTION:   
          for( t = 0; t < a->dimen; t++ ) {
	           
           g95x_traverse_expr( a->start[t], xtc );      
      
           if( a->end[t] ) 
	     g95x_traverse_expr( a->end[t], xtc );        
         
           if( a->stride[t] )        
	     g95x_traverse_expr( a->stride[t], xtc );          
         
          }          
        break;       
       
        case AR_ELEMENT:          
      
          for( t = 0; t < a->dimen; t++ ) 
            g95x_traverse_expr( a->start[t], xtc );    

        break;    
    
        case AR_UNKNOWN: 
	  g95x_die( "Unknown array ref" );
	break; 
 
      }     
      break;
    }
    case REF_COMPONENT: break;      
    case REF_SUBSTRING: 
      g95x_traverse_expr( d->u.ss.start, xtc );          
      g95x_traverse_expr( d->u.ss.end, xtc );          
    break;     
    
    case REF_COARRAY: {
        int t;
        g95_coarray_ref* car = &d->u.car;
        for( t = 0; t < car->dimen; t++ )
          g95x_traverse_expr( car->element[t], xtc );          
          
      }
    break;
    
    default:
    
    g95x_die( "Unknown reference type" );
  }      

}       


/*
 *  constructor traversal
 */

static void traverse_constructor( 
  g95_constructor* b, 
  g95x_traverse_callback *xtc 
) {      
      
  if( xtc->constructor_callback ) {
    if( xtc->constructor_callback( b, xtc->constructor_data ) < 0 ) 
      return;
  }

  if( b->iterator ) {
    g95x_traverse_expr( b->iterator->var, xtc );       
    g95x_traverse_expr( b->iterator->start, xtc );	  
    g95x_traverse_expr( b->iterator->end, xtc ); 	 
    /* Don't traverse artificial step */
    if( g95x_nodummy_expr( b->iterator->step ) ) 
      g95x_traverse_expr( b->iterator->step, xtc );
  }

  g95x_traverse_expr( b->expr, xtc );  
    
}       


/*
 * expr traversal
 */

void g95x_traverse_typespec( g95_typespec *ts, g95x_traverse_callback *xtc ) {
  if( ts->type == BT_CHARACTER ) {
    g95_charlen *cl = ts->cl;
    g95_expr *len;
    if( cl->x.alias ) cl = cl->x.alias;
    len = cl->length;
    if( len ) {
      if( len && len->x.full ) len = len->x.full;
      if( g95x_nodummy_expr( len ) ) /* because parameters
        strings have a length even when declared *(*) */
        g95x_traverse_expr( len, xtc );
    }
  }
  
  if( ts->x.kind )
    g95x_traverse_expr( ts->x.kind->kind, xtc );   
}


void g95x_traverse_expr( g95_expr* h, g95x_traverse_callback *xtc ) {         
  g95_symbol *sy;       
       
  if( h == NULL )       
    return;    

  if( xtc->expr_callback ) {
    if( xtc->expr_callback( h, xtc->expr_data ) < 0 ) 
      return;
  }

  if( h->x.full ) 
    g95x_die( "Should not traverse expr with x.full not null\n" );
    
  switch( h->type ) { 
    case EXPR_SUBSTRING: {
      g95x_traverse_expr( h->ref->u.ss.start, xtc );          
      g95x_traverse_expr( h->ref->u.ss.end, xtc );  
      break;        
    }     
    case EXPR_STRUCTURE: {    
      g95_constructor *c;
      for( c = h->value.constructor.c; c; c = c->next ) {
        g95_expr *e = c->expr;
        if( c->x.dummy )
          continue;
        e = e->x.full ? e->x.full : e;
        g95x_traverse_expr( c->expr, xtc );
      }
      break;
    } 
    case EXPR_ARRAY: {        
      g95_constructor *c;
      g95_ref *r;
      
      for( c = h->value.constructor.c; c; c = c->next )
        traverse_constructor( c, xtc ); 
    
      for( r = h->ref; r; r = r->next )        
        traverse_ref( r, xtc );        

      if( h->x.has_ts ) 
        g95x_traverse_typespec( &h->ts, xtc );

      break;
    }   
    case EXPR_NULL: break;      
    case EXPR_CONSTANT:  
      if( h->ts.x.kind )
        g95x_traverse_expr( h->ts.x.kind->kind, xtc );
    break;     
    case EXPR_VARIABLE: {  
      g95_ref *r;
  
      sy = h->symbol;    
    
      for( r = h->ref; r; r = r->next )         
        traverse_ref( r, xtc );        
      break;
    }
    case EXPR_OP:         
    
      g95x_traverse_expr( h->value.op.op1, xtc );     
     
      if( h->value.op.op2 )       
        g95x_traverse_expr( h->value.op.op2, xtc );         
        
      break;

    case EXPR_FUNCTION: { 
      g95_actual_arglist *k = h->value.function.actual;  
      for( ; k; k = k->next ) 
        if( k->u.expr != NULL )  
          g95x_traverse_expr( k->u.expr, xtc ); 
      if( h->value.function.pointer ) 
        g95x_traverse_expr( h->value.function.pointer, xtc );
      break;      
    }    
    case EXPR_PROCEDURE: 
    break;
    
    default:  
      g95x_die( "Unknown expr type" );        
    break;   
   
  }  

  
}


/*
 * traverse component
 */

static void traverse_component( 
  g95_component *c, g95x_traverse_callback *xtc 
) {
  int i;

  if( c == NULL )    
    return; 
    
  if( xtc->component_callback )
    if( xtc->component_callback( c, xtc->component_data ) < 0 )
      return;

  if( c->initializer ) {
    g95_expr* init = c->initializer;
    if( init && init->x.full ) init = init->x.full;
    g95x_traverse_expr( init, xtc );
  }

  if( c->as ) { 
    g95_array_spec *as = c->as;  
    if( as->x.alias ) 
      as = (g95_array_spec*)as->x.alias;     
    for( i = 0; i < as->rank; i++ ) {
      g95_expr *lower  = as->lower[i];
      g95_expr *upper  = as->upper[i];
      if( lower && lower->x.full ) lower = lower->x.full;
      if( upper && upper->x.full ) upper = upper->x.full;
      if( g95x_nodummy_expr( lower ) ) 
        g95x_traverse_expr( lower, xtc );
      if( g95x_nodummy_expr( upper ) ) 
        g95x_traverse_expr( upper, xtc );
      
    }
  }    

  if( c->ts.x.kind )
    g95x_traverse_expr( c->ts.x.kind->kind, xtc );

  if( c->ts.cl ) {
    g95_expr *length = c->ts.cl->length;
    if( length && length->x.full ) length = length->x.full;
    if( g95x_nodummy_expr( length ) ) /* because component
      strings can be declared character :: s */
    g95x_traverse_expr( length, xtc );
  }

}

void g95x_traverse_array_spec( g95_array_spec *as, g95x_traverse_callback *xtc ) {
  int i;
  if( as ) { 
    if( as->x.alias ) 
      as = (g95_array_spec*)as->x.alias;       
    for( i = 0; i < as->rank; i++ ) {
      g95_expr *lower = as->lower[i];
      g95_expr *upper = as->upper[i];
      if( lower && lower->x.full ) lower = lower->x.full;
      if( upper && upper->x.full ) upper = upper->x.full;
      if( g95x_nodummy_expr( lower ) ) 
        g95x_traverse_expr( lower, xtc );
      if( g95x_nodummy_expr( upper ) ) 
        g95x_traverse_expr( upper, xtc );
    }
  }    
}

void g95x_traverse_coarray_spec( g95_coarray_spec *cas, g95x_traverse_callback *xtc ) {
  int i;
  if( cas ) { 
    if( cas->x.alias ) 
      cas = (g95_coarray_spec*)cas->x.alias;       
    for( i = 0; i < cas->rank; i++ ) {
      g95_expr *lower = cas->lower[i];
      g95_expr *upper = cas->upper[i];
      if( lower && lower->x.full ) lower = lower->x.full;
      if( upper && upper->x.full ) upper = upper->x.full;
      if( g95x_nodummy_expr( lower ) ) 
        g95x_traverse_expr( lower, xtc );
      if( g95x_nodummy_expr( upper ) ) 
        g95x_traverse_expr( upper, xtc );
    }
  }    
}

/*
 *  symbol traversal
 */


void g95x_traverse_symbol( g95_symbol *symbol, g95x_traverse_callback *xtc ) {
  g95_component *k;  
  
         
  if( symbol == NULL )    
    return; 

  if( xtc->symbol_callback )
    if( xtc->symbol_callback( symbol, xtc->symbol_data ) < 0 )
      return;
      
  if( symbol->x.imported ) 
     return;
             
  if( symbol->x.bind ) {
    g95_expr *xbind = symbol->x.bind;
    if( xbind && xbind->x.full ) xbind = xbind->x.full;
  
    if( g95x_nodummy_expr( xbind ) )
      g95x_traverse_expr( xbind, xtc );
  
  }
    
  if( symbol->value ) { 
    g95_expr *value = symbol->value;
    if( value && value->x.full ) value = value->x.full;  
    if( g95x_nodummy_expr( value ) ) /* enumerators may have implicit values */
      g95x_traverse_expr( value, xtc );
  }  
  
  g95x_traverse_typespec( &symbol->ts, xtc );
       
  if( symbol->as ) 
    g95x_traverse_array_spec( symbol->as, xtc );
    
  if( symbol->cas ) 
    g95x_traverse_coarray_spec( symbol->cas, xtc );
    
  if( symbol->components )    
    for( k = symbol->components; k; k = k->next )  
      traverse_component( k, xtc );      
       
}


/*
 * traverse code
 */

void g95x_traverse_code( g95x_statement *sta, g95x_traverse_callback *xtc ) {
  g95_forall_iterator *fa;     
  g95_open *open;      
  g95_alloc *b;  
  g95_code *x;      
  g95_close *close;    
  g95_filepos *f;      
  g95_inquire *s;    
  g95_dt *dt;        
  g95_code *p, code;


  if( ! sta->code )
    return;
        
  code = *(sta->code);
  p = &code;

  if( xtc->code_callback ) {
    if( xtc->code_callback( sta, xtc->code_data ) < 0 ) 
      return;
  }
 
  if( sta->type == ST_NULLIFY ) {
    p->expr2 = NULL;
  }
  
  /* computed goto */
  if( ( sta->type == ST_GOTO ) && ( p->type == EXEC_SELECT ) ) {
    g95x_traverse_expr( p->expr2, xtc );
    return;
  }
 
  switch( p->type ) { 
    case EXEC_NOP:          
    case EXEC_CONTINUE: break;    
      
    case EXEC_AC_ASSIGN:          
      g95x_traverse_expr( p->expr, xtc );
    break;       
         
    case EXEC_ASSIGN:
    case EXEC_POINTER_ASSIGN:  
    if( sta->type == ST_NULLIFY ) {
      g95_code* c;
      int n = 1;
      for( c = p; ; c = c->next, n++ ) {
        g95x_traverse_expr( c->expr, xtc );
        if( n == sta->u.nullify.n )
          break;
      }
    } else {
      g95x_traverse_expr( p->expr, xtc );
      g95x_traverse_expr( p->expr2, xtc );
    }
    break;
   
    case EXEC_GOTO: 
      if( p->expr )
        g95x_traverse_expr( p->expr, xtc );
    break;       
         
    case EXEC_CALL: { 
      g95_actual_arglist *k = p->ext.sub.actual;  
      for( ; k; k = k->next ) {
        if( k->missing_arg_type )
          continue;
        if( k->type != ARG_ALT_RETURN ) 
          g95x_traverse_expr( k->u.expr, xtc ); 
      } 
      if( p->ext.sub.pointer )
        g95x_traverse_expr( p->ext.sub.pointer, xtc );      
    } break;    
   
    case EXEC_RETURN:        
      g95x_traverse_expr( p->expr, xtc );    
    break;     
       
    case EXEC_STOP:          
      g95x_traverse_expr( p->expr, xtc );      
    break;         
           
    case EXEC_PAUSE:       
      g95x_traverse_expr( p->expr, xtc );   
    break; 
   
    case EXEC_ARITHMETIC_IF:   
      g95x_traverse_expr( p->expr, xtc );          
    break;       
         
    case EXEC_IF:       
      g95x_traverse_expr( p->expr, xtc );          
    break;     
       
    case EXEC_SELECT:          
      x = p->block;        
   
      g95x_traverse_expr( p->expr, xtc );
      g95x_traverse_expr( p->expr2, xtc );
      if( p->ext.case_list ) {
        g95_expr *lo = p->ext.case_list->low;
        g95_expr *hi = p->ext.case_list->high;
	if( lo && lo->x.full ) lo = lo->x.full;
	if( hi && hi->x.full ) hi = hi->x.full;
        g95x_traverse_expr( lo, xtc );
        if( lo != hi )
          g95x_traverse_expr( hi, xtc );
      }
      
    break;   
     
    case EXEC_WHERE:       
           
      if( ( x = p->block ) ) {        
        g95x_traverse_expr( x->expr, xtc );
   
        for( x = x->block; x; x = x->block ) {         
          g95x_traverse_expr( x->expr, xtc ); 
        }    
      
      }
      
    break;
   
   
    case EXEC_FORALL:
      for( fa = p->ext.forall_iterator; fa; fa = fa->next ) {     
        g95x_traverse_expr( fa->var, xtc );   
        g95x_traverse_expr( fa->start, xtc );  
        g95x_traverse_expr( fa->end, xtc );	  
        if( g95x_nodummy_expr( fa->stride ) )    
          g95x_traverse_expr( fa->stride, xtc );       
      }        
          
      g95x_traverse_expr( p->expr, xtc );     
    break;    
          
    case EXEC_DO: 
   
      g95x_traverse_expr( p->ext.iterator->var, xtc );       
      g95x_traverse_expr( p->ext.iterator->start, xtc );      
      g95x_traverse_expr( p->ext.iterator->end, xtc );   
     
      if( g95x_nodummy_expr( p->ext.iterator->step ) )
        g95x_traverse_expr( p->ext.iterator->step, xtc );        
          
      break;  
    
    case EXEC_DO_WHILE:          
      /* Changed transform_while */
      if( g95x_nodummy_expr( p->expr ) )
        g95x_traverse_expr( p->expr, xtc ); 
        
/*
      if( p->block 
      && ( p->block != sta->next->code )
      && ( p->block->type == EXEC_IF ) ) {
        if( ( p->block->expr->type == EXPR_OP ) && ! g95x_nodummy_expr( p->block->expr ) ) 
          g95x_traverse_expr( p->block->expr->value.op.op1, xtc ); 
        else 
          g95x_traverse_expr( p->block->expr, xtc ); 
      }       
*/
      
    break;  
    
    case EXEC_CYCLE:
    case EXEC_EXIT:
      break;        
          
    case EXEC_ALLOCATE:
    case EXEC_DEALLOCATE:      
         
      g95x_traverse_expr( p->expr, xtc );     
     
      for( b = p->ext.alloc_list; b; b = b->next ) {        
        g95x_traverse_expr( b->expr, xtc );        
      }          
            
    break; 
   
    case EXEC_OPEN:
      open = p->ext.open;    
            
      g95x_traverse_expr( open->unit, xtc );  
      g95x_traverse_expr( open->iostat, xtc );	
      g95x_traverse_expr( open->file, xtc ); 
      g95x_traverse_expr( open->status, xtc ); 
      g95x_traverse_expr( open->access, xtc );   
      g95x_traverse_expr( open->form, xtc );  
      g95x_traverse_expr( open->recl, xtc );   
      g95x_traverse_expr( open->blank, xtc );	  
      g95x_traverse_expr( open->position, xtc );         
      g95x_traverse_expr( open->action, xtc ); 
      g95x_traverse_expr( open->delim, xtc );  
      g95x_traverse_expr( open->pad, xtc ); 
   
    break;        
          
    case EXEC_CLOSE:  
      close = p->ext.close;  
         
      g95x_traverse_expr( close->unit, xtc );   
      g95x_traverse_expr( close->iostat, xtc );
      g95x_traverse_expr( close->status, xtc );	 
         
    break;        
          
    case EXEC_BACKSPACE:    
    case EXEC_ENDFILE:  
    case EXEC_REWIND:     
      f = p->ext.filepos;   
            
      g95x_traverse_expr( f->unit, xtc ); 
      g95x_traverse_expr( f->iostat, xtc );	     
        
    break;          
            
    case EXEC_INQUIRE:  
      s = p->ext.inquire;   
    
      g95x_traverse_expr( s->unit, xtc ); 
      g95x_traverse_expr( s->file, xtc );        
      g95x_traverse_expr( s->iostat, xtc );	 
      g95x_traverse_expr( s->exist, xtc );   
      g95x_traverse_expr( s->opened, xtc );
      g95x_traverse_expr( s->number, xtc ); 
      g95x_traverse_expr( s->named, xtc );
      g95x_traverse_expr( s->name, xtc );    
      g95x_traverse_expr( s->access, xtc );
      g95x_traverse_expr( s->sequential, xtc );
      g95x_traverse_expr( s->direct, xtc ); 
      g95x_traverse_expr( s->form, xtc );        
      g95x_traverse_expr( s->formatted, xtc );
      g95x_traverse_expr( s->unformatted, xtc );    
      g95x_traverse_expr( s->recl, xtc ); 	   
      g95x_traverse_expr( s->nextrec, xtc ); 
      g95x_traverse_expr( s->blank, xtc ); 
      g95x_traverse_expr( s->position, xtc ); 
      g95x_traverse_expr( s->action, xtc );	   
      g95x_traverse_expr( s->read, xtc );  
      g95x_traverse_expr( s->write, xtc ); 
      g95x_traverse_expr( s->readwrite, xtc );	
      g95x_traverse_expr( s->delim, xtc );      
      g95x_traverse_expr( s->pad, xtc ); 
      g95x_traverse_expr( s->pos, xtc ); 
      g95x_traverse_expr( s->iolength, xtc ); 
      g95x_traverse_expr( s->size, xtc ); 
          
    break;         
           
    case EXEC_IOLENGTH:        
      g95x_traverse_expr( p->expr, xtc );
      goto traverse_transfers;   
        
    case EXEC_READ:       
    case EXEC_WRITE: {
      g95_code *c;    
      dt = p->ext.dt;
           
      if( g95x_nodummy_expr( dt->io_unit ) ) 
        g95x_traverse_expr( dt->io_unit, xtc );   
           
      if( g95x_nodummy_expr( dt->format_expr ) )          
        g95x_traverse_expr( dt->format_expr, xtc );
   
      if( g95x_nodummy_expr( dt->iostat ) )   
        g95x_traverse_expr( dt->iostat, xtc );       
           
      if( g95x_nodummy_expr( dt->size ) )  
        g95x_traverse_expr( dt->size, xtc );   
         
      if( g95x_nodummy_expr( dt->rec ) )  
        g95x_traverse_expr( dt->rec, xtc );        
      
      if( g95x_nodummy_expr( dt->advance ) ) 
        g95x_traverse_expr( dt->advance, xtc );  

      if( g95x_nodummy_expr( dt->pos ) ) 
        g95x_traverse_expr( dt->pos, xtc );  

      if( g95x_nodummy_expr( dt->decimal ) ) 
        g95x_traverse_expr( dt->decimal, xtc );  


traverse_transfers:
      for( c = p->next; c; c = c->next ) {
        switch( c->type ) {
          case EXEC_TRANSFER:
            g95x_traverse_expr( c->expr, xtc );
          break;
          case EXEC_DO:
            g95x_traverse_io_var( c, xtc );
          break;
	  default:
	  break;
        }
        if( c->type == EXEC_DT_END ) 
          break;
      }
    }
    break;
    
    case EXEC_FLUSH: {
      g95_flush* f = p->ext.flush;
      if( g95x_nodummy_expr( f->unit ) ) 
        g95x_traverse_expr( f->unit, xtc );   
      if( g95x_nodummy_expr( f->iostat ) ) 
        g95x_traverse_expr( f->iostat, xtc );   
      if( g95x_nodummy_expr( f->iomsg ) ) 
        g95x_traverse_expr( f->iomsg, xtc );   
    }
    break;
       
    case EXEC_WAIT: {
      g95_wait* f = p->ext.wait;
      if( g95x_nodummy_expr( f->unit ) ) 
        g95x_traverse_expr( f->unit, xtc );   
      if( g95x_nodummy_expr( f->id ) ) 
        g95x_traverse_expr( f->id, xtc );   
      if( g95x_nodummy_expr( f->iostat ) ) 
        g95x_traverse_expr( f->iostat, xtc );   
      if( g95x_nodummy_expr( f->iomsg ) ) 
        g95x_traverse_expr( f->iomsg, xtc );   
    }
    break;
       
    case EXEC_TRANSFER:         
    case EXEC_DT_END: 
      g95x_die( "Attempt to traverse TRANSFER/DT_END code\n" );
    break;
   
    case EXEC_ENTRY: 
    break; 
   
    case EXEC_LABEL_ASSIGN:
      g95x_traverse_expr( p->expr, xtc );          
    break;   


    case EXEC_SYNC_ALL:
    case EXEC_SYNC_TEAM:
    case EXEC_SYNC_IMAGES:
    case EXEC_SYNC_MEMORY:
      if( g95x_nodummy_expr( p->expr ) ) 
        g95x_traverse_expr( p->expr, xtc );
      if( g95x_nodummy_expr( p->ext.sync.stat ) ) 
        g95x_traverse_expr( p->ext.sync.stat, xtc );
      if( g95x_nodummy_expr( p->ext.sync.errmsg ) ) 
        g95x_traverse_expr( p->ext.sync.errmsg, xtc );
    break;  
            
    case EXEC_CRITICAL:
    case EXEC_ALL_STOP:
    break;

    case EXEC_AC_START: /* Should not happen, only defined in scalarize.c */
    case EXEC_WHERE_ASSIGN:

    default:
     g95x_die( "Unknown code" );
    break; 
  }          
          
  
}


/*
 *
 */

void g95x_traverse_data_var( 
  g95_data_variable *var, 
  g95x_traverse_callback *xtc 
) {
  g95_data_variable *v;

  if( var == NULL )    
    return; 
    
  if( xtc->data_var_callback )
    if( xtc->data_var_callback( var, xtc->data_var_data ) < 0 )
      return;
  

  if( var->expr ) 
    g95x_traverse_expr( var->expr, xtc );

  if( var->list ) {
    g95x_traverse_expr( var->iter.var, xtc );
    g95x_traverse_expr( var->iter.start->x.full, xtc );
    g95x_traverse_expr( var->iter.end->x.full, xtc );
    if( g95x_nodummy_expr( var->iter.step->x.full ) ) 
      g95x_traverse_expr( var->iter.step->x.full, xtc );
    for( v = var->list; v; v = v->next ) {
      g95x_traverse_data_var( v, xtc );
    }
  }
  
    

}

/*
 * here, we traverse EXEC_DO codes that are part of a DT_TRANSFER
 * code
 */

void g95x_traverse_io_var( g95_code *c, g95x_traverse_callback* xtc ) {
  g95_code *d;
  
  
  if( c == NULL )    
    return; 
    
  if( xtc->io_var_callback )
    if( xtc->io_var_callback( c, xtc->io_var_data ) < 0 )
      return;

  for( d = c->block; d; d = d->next ) {
    switch( d->type ) {
      case EXEC_DO:
        g95x_traverse_io_var( d, xtc );
      break;
      case EXEC_TRANSFER:
        g95x_traverse_expr( d->expr, xtc );
      break;
      default:
      break;
    }
  }

  g95x_traverse_expr( c->ext.iterator->var, xtc );
  g95x_traverse_expr( c->ext.iterator->start, xtc );
  g95x_traverse_expr( c->ext.iterator->end, xtc );
  if( g95x_nodummy_expr( c->ext.iterator->step ) )    
    g95x_traverse_expr( c->ext.iterator->step, xtc );
  
  
}

