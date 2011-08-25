#include "g95.h"
#include <ctype.h>

#define char_line( ci ) ( (ci)->lb ? (ci)->lb->line[(ci)->column-1] : 0 )
#define char_mask( ci ) ( (ci)->lb ? (ci)->lb->mask[(ci)->column-1] : 0 )
#define char_lt( ci1, ci2 )                 \
( ( (ci1)->lb->rank < (ci2)->lb->rank ) ||  \
  ( ( (ci1)->lb == (ci2)->lb ) &&           \
    ( (ci1)->column < (ci2)->column ) ) )

#define char_le( ci1, ci2 )                  \
( ( (ci1)->lb->rank <= (ci2)->lb->rank ) ||  \
  ( ( (ci1)->lb == (ci2)->lb ) &&            \
    ( (ci1)->column <= (ci2)->column ) ) )

#define isnamec( c ) ( ( ( '0' <= ( c ) ) && ( ( c ) <= '9' ) ) \
                    || ( ( 'a' <= ( c ) ) && ( ( c ) <= 'z' ) ) \
                    || ( ( 'A' <= ( c ) ) && ( ( c ) <= 'Z' ) ) \
                    || ( ( '_' == ( c ) ) ) )

#define isaz( c )    ( ( 'a' <= ( c ) ) && ( ( c ) <= 'z' ) )

#define normalize( xw ) \
  do {                                      \
    if( (xw)->column1 > (xw)->lb1->size+1 ) \
      (xw)->column1 = (xw)->lb1->size+1;    \
    if( (xw)->column2 > (xw)->lb2->size+1 ) \
      (xw)->column2 = (xw)->lb2->size+1;    \
  } while( 0 )

#define check( xw ) \
  do {                                                    \
    if( ( (xw)->lb1 == NULL ) || ( (xw)->column1 == 0 )   \
     || ( (xw)->lb2 == NULL ) || ( (xw)->column2 == 0 )   \
    )                                                     \
    g95x_die( "" );                                       \
  } while( 0 )

#define flat( xw ) \
  ( ( (xw)->lb1 == (xw)->lb2 ) && ( (xw)->column1 == (xw)->column2 ) )

char
g95x_char_move (g95x_char_index * ci, int stop, int step)
{

  if (ci->column > ci->lb->size + 1)
    ci->column = ci->lb->size + 1;

  while (1)
    {

      ci->column += step;
      if (ci->column > ci->lb->size)
	{
	  ci->lb = ci->lb->next;
	  if (ci->lb == NULL)
	    goto out_of_bounds;
	  ci->column = 1;
	}
      else if (ci->column <= 0)
	{
	  g95_linebuf *lb;
	  for (lb = g95_first_line (); lb; lb = lb->next)
	    if (lb->next == ci->lb)
	      break;
	  ci->lb = lb;
	  if (lb == NULL)
	    goto out_of_bounds;
	  ci->column = lb->size;
	}

      if (ci->column >= 1)
        if (ci->lb->mask[ci->column - 1] & stop)
          break;
    }

  return ci->lb->line[ci->column - 1];

out_of_bounds:
  return '\0';
}

void
g95x_locate_expr_nokind (g95x_locus * xw, g95_expr * e)
{
  g95_expr *k;
  g95x_char_index ci;
  char c;
  int step;

  k = e->ts.x.kind ? e->ts.x.kind->kind : NULL;

  if (k == NULL)
    return;

//printf( " l1 = %d, c1 = %d, l2 = %d, c2 = %d\n", 
//  k->x.where.lb1->linenum, k->x.where.column1, 
//  k->x.where.lb2->linenum, k->x.where.column2 );

  if (e->ts.type == BT_CHARACTER)
    {
      step = +1;
      ci.column = k->x.where.column2;
      ci.lb = k->x.where.lb2;
      c = g95x_char_move (&ci, G95X_CHAR_CODE | G95X_CHAR_STRG, -1);
    }
  else
    {
      step = -1;
      ci.column = k->x.where.column1;
      ci.lb = k->x.where.lb1;
    }

  c = g95x_char_move (&ci, G95X_CHAR_CODE | G95X_CHAR_STRG, step);
  if (c != '_')
    g95x_die (" c = `%c' ", c);

  if (step > 0)
    c = g95x_char_move (&ci, G95X_CHAR_CODE | G95X_CHAR_STRG, step);

  if (step < 0)
    {
//  printf( 
//    "\n\n c = '%c' c1 = %d l1 = %d c2 = %d l2 = %d\n\n", 
//    c, e->x.where.column1, e->x.where.lb1->linenum, ci.column, ci.lb->linenum
//  );
      xw->column1 = e->x.where.column1;
      xw->lb1 = e->x.where.lb1;
      xw->column2 = ci.column;
      xw->lb2 = ci.lb;
    }
  else
    {
//  printf( 
//    "\n\n c = '%c' c1 = %d l1 = %d c2 = %d l2 = %d\n\n", 
//    c, ci.column, ci.lb->linenum, e->x.where.column2, e->x.where.lb2->linenum
//  );
      xw->column1 = ci.column;
      xw->lb1 = ci.lb;
      xw->column2 = e->x.where.column2;
      xw->lb2 = e->x.where.lb2;
    }

}



void
g95x_locate_name (g95x_locus * xw)
{
  char name[G95_MAX_SYMBOL_LEN + 1];
  g95x_get_name_substr (xw, name);
}

char *
g95x_get_code_substr (g95x_locus * xw, char *string, int length)
{
  int i = 0;
  g95x_char_index ci1, ci2;


  if ((!xw->lb1) || (xw->column1 == -1) || (xw->column2 == -1))
    g95x_die ("Attempt to take substring of an empty locus\n");

  ci1.lb = xw->lb1;
  ci1.column = xw->column1;
  ci2.lb = xw->lb2;
  ci2.column = xw->column2;

  if (!(char_mask (&ci1) & G95X_CHAR_CODE))
    g95x_char_move (&ci1, G95X_CHAR_CODE, +1);

  while ((char_mask (&ci1) & G95X_CHAR_CODE) && (char_lt (&ci1, &ci2)))
    {
      string[i++] = char_line (&ci1);
      g95x_char_move (&ci1, G95X_CHAR_CODE, +1);
    }

  string[i] = '\0';
  return string;
}

char *
g95x_get_name_substr (g95x_locus * xw, char *name)
{
  int i = 0;
  g95x_char_index ci1, ci2;
  char c;


  if ((!xw->lb1) || (xw->column1 == -1) || (xw->column2 == -1))
    g95x_die ("Attempt to take substring of an empty locus\n");

  ci1.lb = xw->lb1;
  ci1.column = xw->column1;
  ci2.lb = xw->lb2;
  ci2.column = xw->column2;

again:
  if (char_mask (&ci1) != G95X_CHAR_CODE)
    g95x_char_move (&ci1, G95X_CHAR_CODE, +1);

  if (!char_lt (&ci1, &ci2))
    g95x_die ("");

  if (!isnamec (char_line (&ci1)))
    goto again;

  xw->lb1 = ci1.lb;
  xw->column1 = ci1.column;
  while ((char_mask (&ci1) & G95X_CHAR_CODE)
	 && (char_lt (&ci1, &ci2)) && (isnamec (char_line (&ci1))))
    {
      c = char_line (&ci1);
      name[i++] = tolower (c);
      c = g95x_char_move (&ci1, G95X_CHAR_CODE, +1);
    }
  if (ci1.lb != NULL)
    {
      xw->lb2 = ci1.lb;
      xw->column2 = ci1.column;
    }
  else
    {
//    printf ("%d ... %d\n", xw->column2, ci1.column);
    }

  name[i] = '\0';
  return name;
}

void
g95x_refine_location1 (g95x_locus * xw, g95_linebuf ** lb, int *column)
{
  g95x_char_index ci;
  int mask = G95X_CHAR_CODE | G95X_CHAR_STRG;

  check (xw);
  normalize (xw);

  *lb = ci.lb = xw->lb1;
  *column = ci.column = xw->column1;

  if (flat (xw))
    return;

  if (char_mask (&ci) & mask)
    return;

  g95x_char_move (&ci, mask, 1);
  *lb = ci.lb;
  *column = ci.column;

}


void
g95x_refine_location2 (g95x_locus * xw, g95_linebuf ** lb, int *column)
{
  g95x_char_index ci;
  int mask = G95X_CHAR_CODE | G95X_CHAR_STRG;

  check (xw);
  normalize (xw);

  *lb = ci.lb = xw->lb2;
  *column = ci.column = xw->column2;

  if (flat (xw))
    return;

  g95x_char_move (&ci, mask, -1);
  *lb = ci.lb;
  *column = ci.column + 1;

}

void
g95x_refine_location (g95x_locus * xw, g95x_locus ** pxr, int *pnxr)
{
  static g95x_locus xr[256];
  g95_linebuf *lb;
  int l;
  int ixr = 0;

  if (xw->lb1 == NULL)
    g95x_die ("");

  l = xw->lb1->linenum;

  for (lb = xw->lb1; lb; lb = lb->next, l++)
    {
      short *mask = lb->mask;
      int c1 = 1, c2 = lb->size + 1, c, d1 = -1, d2 = -1;

      if (lb == xw->lb1)
	c1 = xw->column1;
      if ((lb == xw->lb2) && (xw->column2 < c2))
	c2 = xw->column2;

      for (c = c1; c < c2; c++)
	{
	  if ((d1 < 0) && ((mask[c - 1] == G95X_CHAR_CODE)
			   || (mask[c - 1] == G95X_CHAR_STRG)))
	    {
	      d1 = c;
	    }
	  if ((d1 > 0) && ((mask[c - 1] == G95X_CHAR_CODE)
			   || (mask[c - 1] == G95X_CHAR_STRG)))
	    {
	      d2 = c;
	    }

	}

      if (d1 > 0)
	{
	  xr[ixr].lb1 = lb;
	  xr[ixr].column1 = d1;
	  xr[ixr].lb2 = lb;
	  xr[ixr].column2 = d2 + 1;
	  ixr++;
	}

      if (lb == xw->lb2)
	break;

    }

  *pnxr = ixr;
  *pxr = &xr[0];

}

int
g95x_compare_locus (const g95x_locus * xwa, const g95x_locus * xwb)
{
  if (xwa->lb1->rank < xwb->lb1->rank)
    return -1;
  if (xwa->lb1->rank > xwb->lb1->rank)
    return +1;
  if (xwa->column1 < xwb->column1)
    return -1;
  if (xwa->column1 > xwb->column1)
    return +1;
  return 0;
}

int
g95x_opl (const g95x_locus * xw, g95x_locus * xwo, const char *operator1,
	  const char *operator2)
{
  g95x_char_index ci1, ci2;
  char c1, c2;
  int n1 = strlen (operator1), n2 = strlen (operator2);

  ci1.lb = xw->lb1;
  ci1.column = xw->column1;

  ci2.lb = xw->lb2;
  ci2.column = xw->column2;

  *xwo = *xw;

  while (n1--)
    {
      c1 = g95x_char_move (&ci1, G95X_CHAR_CODE, -1);
      if (tolower (c1) != tolower (operator1[n1]))
	g95x_die ("");
    }

  if (char_mask (&ci2) != G95X_CHAR_CODE)
    c2 = g95x_char_move (&ci2, G95X_CHAR_CODE, +1);
  c2 = char_line (&ci2);
  while (n2--)
    {
      if (tolower (c2) != tolower (operator2[n2]))
	g95x_die ("");
      c2 = g95x_char_move (&ci2, G95X_CHAR_CODE, +1);
    }

  xwo->lb1 = ci1.lb;
  xwo->column1 = ci1.column;
  xwo->lb2 = ci2.lb;
  xwo->column2 = ci2.column;

  return 0;
}

int
g95x_iok (g95x_locus * xw, const char *kwrd, g95x_locus * xwo)
{
  g95x_char_index ci;
  ci.lb = xw->lb1;
  ci.column = xw->column1;
  char c;
  int n = strlen (kwrd);

  *xwo = *xw;

  c = g95x_char_move (&ci, G95X_CHAR_CODE, -1);
  if ((c == ',') || (c == '('))
    return 0;

  if (c != '=')
    {
      if (strcmp ("fmt", kwrd))
	g95x_die ("");
      else
	return 0;
    }

  while (n--)
    {
      c = g95x_char_move (&ci, G95X_CHAR_CODE, -1);
      if (tolower (c) != tolower (kwrd[n]))
	g95x_die ("");
    }

  xwo->lb1 = ci.lb;
  xwo->column1 = ci.column;

  c = g95x_char_move (&ci, G95X_CHAR_CODE, -1);
  if ((c == ',') || (c == '('))
    return 0;

  g95x_die ("");

  return 0;
}


static void
xw_lang (g95x_locus * xw, int k, g95x_statement * sta)
{
  *xw = sta->where;

  if (k == 0)
    {
      xw->lb1 = sta->where.lb1;
      xw->column1 = sta->where.column1;
      xw->lb2 = sta->ext_locs[k].loc.lb1;
      xw->column2 = sta->ext_locs[k].loc.column1;
    }
  else if (k < sta->n_ext_locs)
    {
      xw->lb1 = sta->ext_locs[k - 1].loc.lb2;
      xw->column1 = sta->ext_locs[k - 1].loc.column2;
      xw->lb2 = sta->ext_locs[k].loc.lb1;
      xw->column2 = sta->ext_locs[k].loc.column1;
    }
  else if (k == sta->n_ext_locs)
    {
      xw->lb1 = sta->ext_locs[k - 1].loc.lb2;
      xw->column1 = sta->ext_locs[k - 1].loc.column2;
      xw->lb2 = sta->where.lb2;
      xw->column2 = sta->where.column2;
    }
  else
    {
      g95x_die ("");
    }
}

static void
get_substr (g95x_locus * xw, char **pstr, g95x_char_index ** pci)
{
  static char str[1024];
  static g95x_char_index str_ci[1024];
  int n = 0;
  g95x_char_index ci1, ci2;
  char c;

  ci1.lb = xw->lb1;
  ci1.column = xw->column1;
  ci2.lb = xw->lb2;
  ci2.column = xw->column2;

  if (!(char_mask (&ci1) & G95X_CHAR_CODE))
    g95x_char_move (&ci1, G95X_CHAR_CODE, 1);
  while (char_lt (&ci1, &ci2))
    {
      c = char_line (&ci1);
      str_ci[n] = ci1;
      str[n] = tolower (c);
      g95x_char_move (&ci1, G95X_CHAR_CODE, 1);
      n++;
    }
  str[n] = '\0';

  *pstr = &str[0];
  *pci = &str_ci[0];
}

int
g95x_match_word (g95x_locus * xw, const char *word)
{
  g95x_char_index ci1, ci2;
  int n = 0;
  char *str;
  static g95x_char_index *str_ci;
  char *p;


  ci1.lb = xw->lb1;
  ci1.column = xw->column1;
  ci2.lb = xw->lb2;
  ci2.column = xw->column2;

  get_substr (xw, &str, &str_ci);

  n = strlen (str);
  if ((p = strstr (str, word)))
    {
      int k = p - str;
      ci1 = str_ci[k];
      k += strlen (word);
      xw->lb1 = ci1.lb;
      xw->column1 = ci1.column;
      if (k < n)
	{
	  ci2 = str_ci[k];
	  xw->lb2 = ci2.lb;
	  xw->column2 = ci2.column;
	}
      return 1;
    }
  return 0;
}

int
g95x_locate_attr (g95x_statement * sta, const char *name, g95x_locus * xwa)
{
  int k;

  for (k = 0; k <= sta->n_ext_locs; k++)
    {
      xw_lang (xwa, k, sta);
      if (g95x_match_word (xwa, name))
	return 0;
    }

  g95x_die ("");

  return 0;
}

int
g95x_parse_letter_spec_list (g95x_locus * xw, int *pk, g95x_letter_spec * c)
{
  char *str;
  g95x_char_index *ci;
  int i, n, k;

  for (k = 0; k < 26; k++)
    {
      c[k].c1 = c[k].c2 = '\0';
    }

  get_substr (xw, &str, &ci);
  n = strlen (str);
//printf( " str = >%s< \n", str );


  if (str[0] != '(')
    g95x_die ("");
  if (str[n - 1] != ')')
    g95x_die ("");

  k = 0;
  i = 1;

  while (i < n - 1)
    {
      if (!isaz (str[i]))
	g95x_die ("");
      c[k].c1 = str[i];
      c[k].where1.lb1 = ci[i].lb;
      c[k].where1.column1 = ci[i].column;
      i++;
      k++;
      if (!(i < n - 1))
	break;
      if (str[i] == ',')
	goto again;
      if (str[i] != '-')
	g95x_die ("");
      i++;
      if (!(i < n - 1))
	g95x_die ("");
      if (!isaz (str[i]))
	g95x_die ("");
      c[k - 1].c2 = str[i];
      c[k - 1].where1.lb2 = ci[i].lb;
      c[k - 1].where1.column2 = ci[i].column;

      i++;
      if (!(i < n - 1))
	break;
      if (str[i] != ',')
	g95x_die ("");
    again:
      i++;
    }

  *pk = k;

  for (k = 0; k < *pk; k++)
    {
      if (c[k].c2 > 0)
	{
	  c[k].where.lb1 = c[k].where1.lb1;
	  c[k].where.column1 = c[k].where1.column1;
	  c[k].where.lb2 = c[k].where1.lb2;
	  c[k].where.column2 = c[k].where1.column2 + 1;
	}
      else
	{
	  c[k].where.lb1 = c[k].where1.lb1;
	  c[k].where.column1 = c[k].where1.column1;
	  c[k].where.lb2 = c[k].where1.lb1;
	  c[k].where.column2 = c[k].where1.column1 + 1;
	}
    }

  return 0;
}
