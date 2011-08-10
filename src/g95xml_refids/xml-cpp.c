#include "g95.h"
#include "xml-cpp.h"

#include "config.h"
#include "system.h"
#include "cpplib.h"
#include "internal.h"



g95x_cpp* g95x_get_cpp( int ss, int st ) {
  g95x_cpp* xcp;
  xcp = (g95x_cpp*)g95_getmem( 
    sizeof( g95x_cpp ) + ss * 2 * sizeof( short int ) + st 
  );
  xcp->n = ss;
  xcp->next = NULL;
  return xcp;
}

static g95x_cpp *xcp_list = NULL, *xcp_cur = NULL;

void g95x_push_cpp( g95x_cpp *xcp ) {
  if( xcp_cur ) {
    xcp_cur->next = xcp;
    xcp_cur = xcp;
  } else {
    xcp_cur = xcp_list = xcp;
  }
}

void g95x_push_cpp_line( struct g95_file *current ) {
  g95x_cpp_line *xcl;

  if( xcp_list == NULL ) 
    return;
  xcl = g95x_get_cpp_line();
  xcl->next = current->x.cl;
  current->x.cl = xcl;
  xcl->xcp = xcp_list;
  xcl->line = g95x_current_line( current );
  xcp_list = xcp_cur = NULL;
}


