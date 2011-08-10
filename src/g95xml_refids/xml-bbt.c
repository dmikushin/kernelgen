#include "g95.h"
#include <string.h>




/*
 * Record all ids printed in a bbt
 */

g95x_bbt_id_root root_bbt_id = { NULL, 0 };


#define COMPARE_BBT( st1, st2 ) \
  (st1)->p < (st2)->p        ? -1                                 : \
  (st1)->p > (st2)->p        ? +1                                 : \
  (st1)->name && (st2)->name ? strcmp( (st1)->name, (st2)->name ) : \
  (st1)->name                ? +1                                 : \
  (st2)->name                ? -1                                 : \
                                0

static int compare_bbt_id( g95x_bbt_id *st1, g95x_bbt_id *st2 ) {
  return COMPARE_BBT( st1, st2 );
}

g95x_bbt_id *g95x_new_bbt_id( g95x_pointer p, char* name, g95x_bbt_id_root* bbt_root ) {
  g95x_bbt_id *st;

  st = g95_getmem( sizeof( g95x_bbt_id ) );
  st->p = p;
  bbt_root->bbt_id_number++;
  st->q = bbt_root->bbt_id_number;
  if( name ) {
    st->name = g95_getmem( strlen( name ) + 1 );
    strcpy( st->name, name );
  } else 
    st->name = NULL;
  

  g95_insert_bbt( &bbt_root->root, st, compare_bbt_id );
  
  return st;
}


g95x_bbt_id *g95x_get_bbt_id( g95x_pointer p, char* name, g95x_bbt_id_root* bbt_root ) {
int c;
g95x_bbt_id* st = bbt_root->root;
g95x_bbt_id st0;
  
  st0.p = p;
  st0.name = name;

  while( st != NULL ) {

    c = COMPARE_BBT( &st0, st );

    if( c == 0 ) 
      return st;

    st = c < 0 ? st->left : st->right;
  }

  return NULL;
}


static void traverse_bbt_ids0( g95x_bbt_id *st, void (*func)( g95x_bbt_id*, void* ), void* data ) {
  if( ! st )
    return;
  if( st->left )
    traverse_bbt_ids0( st->left, func, data );
  if( st->right )
    traverse_bbt_ids0( st->right, func, data );
  func( st, data );
}

static void traverse_bbt_ids( g95x_bbt_id* root, void (*func)( g95x_bbt_id*, void* ), void* data ) {
  traverse_bbt_ids0( root, func, data );
}

static void free_bbt_id( g95x_bbt_id* st, void* data ) {
  if( st->name )
    g95_free( st->name );
  g95_free( st );
}

void g95x_free_bbt_ids( g95x_bbt_id_root *bbt_root ) {
  traverse_bbt_ids( bbt_root->root, free_bbt_id, NULL );
}

static void check_bbt_id( g95x_bbt_id* st, void* data ) {
  int* perr = (int*)data;
  int i;
  if( st->defined == 0 ) {
    (*perr)++;
    fprintf( stderr, "Undefined reference " );
    if( g95x_option.canonic ) {
      fprintf( stderr, REFFMT0, st->q );
    } else {
      fprintf( stderr, REFFMT0, st->p );
      if( st->name ) {
        fprintf( stderr, "x" );
        for( i = 0; st->name[i]; i++ )  
          fprintf( stderr, "%2.2x", st->name[i] );
      }
    }
    fprintf( stderr, "\n" );
  }
  if( st->defined > 1 ) {
    (*perr)++;
    fprintf( stderr, "Multiple reference definition " );
    if( g95x_option.canonic ) {
      fprintf( stderr, REFFMT0, st->q );
    } else {
      fprintf( stderr, REFFMT0, st->p );
      if( st->name ) {
        fprintf( stderr, "x" );
        for( i = 0; st->name[i]; i++ )  
          fprintf( stderr, "%2.2x", st->name[i] );
      }
    }
    fprintf( stderr, "\n" );
  }
    
}

void g95x_check_bbt_ids() {
  int err = 0;
  
  traverse_bbt_ids( root_bbt_id.root, check_bbt_id, &err );
  
  if( err > 0 )
    g95x_die( "XML errors = %d\n", err );
}





