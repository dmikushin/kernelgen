#include "config.h"
#include "system.h"
#include "cpplib.h"
#include "internal.h"


const char* progname = "main.ex";

hash_table *ident_hash;
struct line_maps line_table;

static void *alloc_subobject( size_t x ) {
  void *p =  xmalloc( x );
/*  printf( "alloc_subobject = 0x%x\n", p );*/
  return p;
}

static hashnode alloc_node( 
  hash_table *table ATTRIBUTE_UNUSED 
) {
  hashnode hn;
  hn = xmalloc( sizeof( struct cpp_hashnode ) );
/*  printf( "alloc_node = 0x%x\n", hn );*/
  return hn;
}


static void file_change(
  cpp_reader *input, 
  const struct line_map *map
) { 
  if( map == NULL )
    return;         
         
/*
  if( map->reason == LC_ENTER ) {    
    printf( "Entering %s\n", map->to_file );
  } else if( map->reason == LC_LEAVE ) {
    printf( "Leaving %s\n", map->to_file );
  }       
*/

}  


void init() {
  ident_hash = ht_create( 14 );
  ident_hash->alloc_node = alloc_node;
  ident_hash->alloc_subobject = alloc_subobject;
}

char *builtin_macros[] = {
   "__G95__ 0",
   "__G95_MINOR__ 50",
   "__FORTRAN__ 95",
"linux 1",
"__linux__ 1",
"unix 1",
"__unix__ 1",
 NULL };


int main( int argc, char* argv[] ) {
  cpp_reader *input;
  cpp_callbacks *cb; 
  int leng; 
  char v;
  cpp_dir *s;    
  const struct line_map *map;       
  char **r;
  
    
  init();
  
  input = cpp_create_reader( CLK_GNUC89, ident_hash, &line_table );       
  cb = cpp_get_callbacks(input);       
  cb->file_change = file_change;      

  if( cpp_read_main_file( input, argv[1] ) == NULL )
    printf( "FAILURE!\n" ); 

  CPP_OPTION( input, traditional ) = 1; 
  cpp_post_options( input );      
  cpp_set_lang( input, CLK_STDC94 );       
  CPP_OPTION( input, trigraphs ) = 0;          

  cpp_init_builtins( input, 1 );

  for( r = builtin_macros; *r != NULL; r++ )
    _cpp_define_builtin( input, *r );


  s = xmalloc( sizeof( cpp_dir ) );
  s->name = "/home/phi001/fortran/cpp";      
  s->len = strlen( s->name );    
  s->user_supplied_p = true;  
  s->next = NULL;
  
  cpp_set_include_chains( input, s, NULL, 0 );        

  while( _cpp_read_logical_line_trad( input ) ) {  
    map = linemap_lookup( &line_table, input->out.first_line ); 
/*
    current_file->line = source_line( input, map );  
*/
    leng = input->out.cur - input->out.base;     

    v = input->out.base[leng];
    input->out.base[leng] = '\0';       

    printf( "%s\n", input->out.base );     

    input->out.base[leng] = v;       
    

  }  

  cpp_destroy( input );       
    
}









