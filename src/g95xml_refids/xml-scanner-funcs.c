
hash_table *ident_hash;
struct line_maps line_table;

static int init_macros = 0;

static int source_line( 
  cpp_reader *input, const struct line_map *map, source_location n );


/* linebuf allocator */

static g95_linebuf *get_linebuf( int size ) {
  g95_linebuf *k;
  int i;
  k = g95_getmem( sizeof( g95_linebuf ) + 2 * size );
  k->size = size;
  k->mask = (char*)k->line + size + 1;
  for( i = 0; i < size; i++ ) {
    k->mask[i] = G95X_CHAR_NONE;
  }
  k->mask[size] = '\0';
  return k;
}



/* cpp callbacks */

static const char* missing_header( 
  cpp_reader* input, const char* header, cpp_dir **dir 
) {
  FILE *fp = fopen( header, "r" );
  if( fp == NULL )
    g95_fatal_error( 
      "%s:%d: %s: %s", 
      current_name, current_file->line,
      header, strerror( errno )
    );
  fclose( fp );
  return header;
}



static void define( 
  cpp_reader* input, unsigned int k, cpp_hashnode* hnode 
) {
  return;
#ifdef UNDEF
  const struct line_map *map = linemap_lookup( &line_table, input->out.first_line );
  int line = source_line( input, map, input->out.first_line );
  cpp_macro *macro = hnode->value.macro;

  printf( "<cpp_define id=\"0x%x\" macro=\"0x%x\"", hnode, macro );

  if( ! init_macros ) 
    printf( " f=\"0x%x\" line=\"%d\"", current_file, line );
  
  g95x_print( "/>\n" );
  
  g95x_print_macro( hnode );
#endif
}

static void undef( 
  cpp_reader* input, unsigned int k, cpp_hashnode* hnode 
) {
  return;
#ifdef UNDEF
  const struct line_map *map = linemap_lookup( &line_table, input->out.first_line );
  int line = source_line( input, map, input->out.first_line );
  cpp_macro *macro = hnode->value.macro;

  if( ! macro ) 
    return;

  void *m = hnode->flags & NODE_BUILTIN ? hnode : macro;

  printf( "<cpp_undef id=\"0x%x\" macro=\"0x%x\"", hnode, m );
  
  if( ! init_macros )
    printf( " f=\"0x%x\" line=\"%d\"", current_file, line );

  g95x_print( "/>\n" );
#endif
}


/* 
 * cpp helper functions; we have to manage memory allocation
 * for cpp, we do that by allocating ~ 4000b chunks of memory
 */


typedef struct _cpp_page_alloc {
  struct _cpp_page_alloc* next;
  int length;
  int where;
  char mem[1];
} _cpp_page_alloc;

static _cpp_page_alloc* _get_cpp_page_alloc( int length ) {
  _cpp_page_alloc* cpa = g95_getmem( sizeof( struct _cpp_page_alloc ) + length );
  cpa->length = length;
  cpa->where = 0;
  return cpa;
}

static _cpp_page_alloc *_cpp_page_alloc_head = NULL, 
                       *_cpp_page_alloc_tail = NULL;

static void* get_cpp_mem( int size ) {
  void *p = NULL;
  if( _cpp_page_alloc_head == NULL ) {
    _cpp_page_alloc_head =
    _cpp_page_alloc_tail =
    _get_cpp_page_alloc( 4000 );
  }
  if( _cpp_page_alloc_tail->where + size >= _cpp_page_alloc_tail->length ) {
    _cpp_page_alloc_tail->next = _get_cpp_page_alloc( 4000 + size );
    _cpp_page_alloc_tail = _cpp_page_alloc_tail->next;
  }
  p = &_cpp_page_alloc_tail->mem[ _cpp_page_alloc_tail->where ];
  _cpp_page_alloc_tail->where += size;
  return p;
}

static void *alloc_subobject( size_t x ) {
  return get_cpp_mem( x );
}

static hashnode alloc_node( 
  hash_table *table ATTRIBUTE_UNUSED 
) {
  return get_cpp_mem( sizeof( struct cpp_hashnode ) );
}




static void scanner_init() {
  static int done = 0;
  if( done == 0 ) {
    done++;
    ident_hash = ht_create( 14 );
    ident_hash->alloc_node = alloc_node;
    ident_hash->alloc_subobject = alloc_subobject;
  }
}





g95_file* g95x_get_file_head() {
  return file_head;
}



static const int color_blue[]   = { 0x1b, 0x5b, 0x33, 0x34, 0x6d, 0x00 };
static const int color_red[]    = { 0x1b, 0x5b, 0x33, 0x31, 0x6d, 0x00 };
static const int color_yellow[] = { 0x1b, 0x5b, 0x33, 0x33, 0x6d, 0x00 };

static const int color_reset[]  = { 0x1b, 0x5b, 0x30, 0x6d, 0x00 };

#define COLOR( x ) \
  do { int i; for( i = 0; color_##x[i]; i++ ) printf( "%c", color_##x[i] ); } while( 0 )


void g95x_print_mask() {
  g95_linebuf *lb;
  int i;
  int j;
  int n = 70;
  int in_string = 0;
  int in_comment = 0;
  int f;
  
  return;
  
  for( f = 0; f < 1; f++ ) {
  
    for( i = 0; i < n; i++ ) {
      printf( "%d", i % 10 );
    }  
    printf( "\n" );
    for( lb = line_head; lb; lb = lb->next ) {
      for( j = 0; j < lb->size; j++ ) {
        if( ( lb->mask[j] == G95X_CHAR_COMM ) && ! in_comment ) {
          COLOR( blue );
          in_comment++;
        }
        if( in_string && ( lb->mask[j] != G95X_CHAR_STRG ) ) {
          COLOR( reset );
          in_string = 0;
        }
        if( ( lb->mask[j] == G95X_CHAR_STRG ) && ! in_string ) {
          COLOR( red );
          in_string++;
        }
        if( ( lb->mask[j] == G95X_CHAR_CTFX ) 
         || ( lb->mask[j] == G95X_CHAR_CTFR ) 
         || ( lb->mask[j] == G95X_CHAR_ZERO ) ) {
          COLOR( yellow );
        }
        printf( "%c", f == 0 ? lb->line[j] : lb->mask[j] );
        if( ( lb->mask[j] == G95X_CHAR_CTFX ) 
         || ( lb->mask[j] == G95X_CHAR_CTFR ) 
         || ( lb->mask[j] == G95X_CHAR_ZERO ) ) {
          COLOR( reset );
        }
      }
      COLOR( reset );
      in_comment = 0;
      in_string = 0;
      printf( "\n" );
    }

  }
    
}


static void print_comments( g95_file* f ) {
  g95_linebuf *lb;
  int i;
  int b = 0;
  g95x_print( " comments=\"[" );
  for( lb = line_head; lb; lb = lb->next ) {
    if( lb->file != f )
      continue;
    for( i = 0; i < lb->size; i++ ) {
      if( lb->mask[i] == G95X_CHAR_COMM ) {
        if( b++ )
          g95x_print( "," );
        g95x_print( "%d,%d,%d,%d", lb->linenum-1, i, lb->linenum-1, lb->size );
        break;
      }
    }
  }
  g95x_print( "]\"" );
}

static void print_cpp( g95x_cpp *xcp ) {
  int sa;
  for( ; xcp; xcp = xcp->next ) {
    g95x_print( "<cpp id=\"%R\"", xcp );
//    g95x_print( " macro=\"%R\"", xcp->macro );
    g95x_print( " c1=\"%d\" c2=\"%d\" items=\"", xcp->c1, xcp->c2 );
    for( sa = 0; sa < xcp->n; sa++ ) {
      g95x_print( "%R", &xcp->s[sa] );
      if( sa < xcp->n - 1 ) 
        g95x_print( "," );
    }
    g95x_print( "\"/>\n" );
    for( sa = 0; sa < xcp->n; sa++ ) {
      g95x_print( "<cpp_item id=\"%R\"", &xcp->s[sa] );
      if( xcp->s[sa].c1 < 0 ) {
        g95x_print( " text=\"" );
        g95x_print_string( (char*)xcp + xcp->s[sa].c2, strlen( (char*)xcp + xcp->s[sa].c2 ), 0 );
        g95x_print( "\"" );
      } else {
        g95x_print( " c1=\"%d\" c2=\"%d\"", xcp->s[sa].c1, xcp->s[sa].c2 );
      }
      g95x_print( "/>\n" );
    }
    
  }
}

static void print_lines( g95_file *f ) {
  g95_linebuf *lb;
  int p = 0;
  
  g95x_print( " lines=\"" );
  for( lb = line_head; lb; lb = lb->next ) {
    if( lb->file == f ) {
      if( p )
        g95x_print( "," );
      g95x_print( "%d", lb->linenum-1 );
      p = 1; 
    }
  }
  g95x_print( "\"" );
  
}

static void print_ampersands( g95_file * f ) {
  g95_linebuf *lb;
  int p = 0;
  int i;
 
  g95x_print( " ampersands=\"[" );
  for( lb = line_head; lb; lb = lb->next ) {
    if( lb->file == f ) 
      for( i = 0; i < lb->size; i++ ) 
        if( lb->mask[i] == G95X_CHAR_CTFR ) {
          if( p )
            g95x_print( "," );
          g95x_print( "%d,%d", lb->linenum-1, i );
          p = 1;
        }
  }
  g95x_print( "]\"" );
  
  g95x_print( " continuations=\"[" );
  for( lb = line_head; lb; lb = lb->next ) {
    if( lb->file == f ) 
      for( i = 0; i < lb->size; i++ ) 
        if( lb->mask[i] == G95X_CHAR_CTFX ) {
          if( p )
            g95x_print( "," );
          g95x_print( "%d,%d", lb->linenum-1, i );
          p = 1;
        }
  }
  g95x_print( "]\"" );
  
  g95x_print( " zeros=\"[" );
  for( lb = line_head; lb; lb = lb->next ) {
    if( lb->file == f ) 
      for( i = 0; i < lb->size; i++ ) 
        if( lb->mask[i] == G95X_CHAR_ZERO ) {
          if( p )
            g95x_print( "," );
          g95x_print( "%d,%d", lb->linenum-1, i );
          p = 1;
        }
  }
  g95x_print( "]\"" );
  
  
}
static void print_ext_locs( g95_file* f ) {
  g95x_statement *sta = g95x_statement_head;
  int printed = 0;

  if( g95x_option.ext_locs_in_statement )
    return;

  g95x_print( " ext_locs=\"" );
  for( ; sta; sta = sta->a_next ) {

    if( sta->where.lb1->file == f ) {

      if( printed && sta->n_ext_locs )
        g95x_print( "," );

      printed += g95x_print_statement_ext_locs( sta );    

    }
    if( sta == g95x_statement_tail )
      break;  
  }
  g95x_print( "\"" );

}

static void print_file( g95_file* f ) {
  g95x_cpp_line *xcl;
  g95x_print( 
    "<file id=\"%R\" name=\"%s\" nline=\"%d\" width=\"%d\"",
    f, f->filename, f->line, line_width
  );
  if( f->included_by ) 
    g95x_print( 
      " inclusion_line=\"%d\" included_by=\"%R\"", 
      f->inclusion_line - 1, f->included_by 
    );
  switch( f->x.form ) {
    case FORM_FIXED:
      g95x_print( " form=\"FIXED\"" );
    break;
    case FORM_FREE:
      g95x_print( " form=\"FREE\"" );
    break;
    case FORM_UNKNOWN:
      g95x_print( " form=\"UNKNOWN\"" );
    break;
  }
  
  
  print_comments( f );
  print_lines( f );

  print_ext_locs( f );

  print_ampersands( f );

  g95x_print( " cpp_lines=\"" );
  for( xcl = f->x.cl; xcl; xcl = xcl->next ) {
    g95x_print( "%R", xcl );
    if( xcl->next )
      g95x_print( "," );
  }
  g95x_print( "\"" );

  g95x_print( "/>\n" );
  
  for( xcl = f->x.cl; xcl; xcl = xcl->next ) {
    g95x_cpp *xcp;
    g95x_print( "<cpp_line id=\"%R\" line=\"%d\" cpps=\"", xcl, xcl->line-1 );
    for( xcp = xcl->xcp; xcp; xcp = xcp->next ) { 
      g95x_print( "%R", xcp );
      if( xcp->next )
        g95x_print( "," );
    }
    g95x_print( "\"/>\n" );
    print_cpp( xcl->xcp );
  }
  
}

void g95x_print_files() {
  g95_file *f;
  for( f = file_head; f; f = f->next ) 
    print_file( f );
}



int g95x_current_line( g95_file *file ) {
  int line;
  cpp_reader *input;
  const struct line_map *map;
  
  input = file->x.input;
  map = linemap_lookup(
    &line_table, input->out.first_line
  ); 
  
  line = source_line( input, map, input->out.first_line );  

  return line;
}
