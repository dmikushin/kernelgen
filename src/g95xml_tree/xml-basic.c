#include "g95.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

g95_namespace *g95x_dump_ns = NULL;

static char *char_null = NULL;
char **g95x_builtin_macros = &char_null;


g95x_option_t g95x_option = {
    0,        /* int enable;              */
    0,        /* int argc;                */
    NULL,     /* char **argv;             */
    NULL,     /* char *xmlns;             */
    NULL,     /* char *xmlns_uri;         */
    NULL,     /* char *stylesheet;        */
    NULL,     /* char *stylesheet_type;   */
    1,        /* int xml_type;            */
    0,        /* int xml_no_header;       */
    0,        /* int xml_defn;            */
    0,        /* int xml_xul;             */
    0,        /* int xml_sc;              */
    0,        /* int xml_no_ns;           */
    0,        /* int xml_example;         */
    0,        /* int xml_watch;           */
    0         /* int xml_dump_mask;       */
};

const char *g95x_current_intrinsic_module_name = NULL;



static char *
get_opt_eq (char **val, char *arg, const char *opt)
{
  int len = strlen (opt);
  if ((arg[0] == '-')
      && (!strncmp (&arg[1], opt, len)) && (arg[len + 1] == '='))
    {
      *val = &arg[len + 2];
      return *val;
    }
  else
    {
      return NULL;
    }
}


static int
get_opt_f (int *val, char *arg, const char *opt)
{
  if ((arg[0] == '-') && (!strcmp (&arg[1], opt)))
    {
      *val = 1;
      return 1;
    }
  return 0;
}

int
g95x_parse_arg (int argc, char *argv[])
{
  char *macros, *arch;

  if (!strcmp(argv[0], "-o"))
    return 2;

  if (get_opt_f (&g95x_option.enable, argv[0], "xml"))
    return 1;

  if (get_opt_f (&g95x_option.enable, argv[0], "xml-out=-"))
    return 1;

  if (get_opt_f (&g95x_option.enable, argv[0], "c"))
    return 1;

  if (get_opt_f (&g95x_option.xml_xul, argv[0], "xul")) {
    g95x_option.enable = 1;
    return 1;
  }

  if (get_opt_f (&g95x_option.xml_watch, argv[0], "xml-watch"))
    return 1;

  if (get_opt_f (&g95x_option.xml_example, argv[0], "xml-example"))
    return 1;

  if (get_opt_f (&g95x_option.xml_dump_mask, argv[0], "xml-dump-mask"))
    return 1;

  if (get_opt_f (&g95x_option.xml_no_ns, argv[0], "xml-no-ns"))
    return 1;

  if (get_opt_f (&g95x_option.xml_no_header, argv[0], "xml-no-header"))
    return 1;

  if (get_opt_f (&g95x_option.xml_sc, argv[0], "xml-code-sr"))
    return 1;

  if (get_opt_f (&g95x_option.xml_defn, argv[0], "xml-defn"))
    return 1;

  if (get_opt_eq (&g95x_option.xmlns, argv[0], "xml-ns-prefix"))
    return 1;

  if (get_opt_eq (&g95x_option.stylesheet, argv[0], "xml-stylesheet"))
    return 1;

  if (get_opt_eq
      (&g95x_option.stylesheet_type, argv[0], "xml-stylesheet-type"))
    return 1;

  if (get_opt_eq (&g95x_option.xmlns_uri, argv[0], "xml-ns-uri"))
    return 1;


  if (get_opt_eq (&macros, argv[0], "xml-macros"))
    {

/*
char *builtin_macros[] = {
   "__G95__ 0",
   "__G95_MINOR__ 50",
   "__FORTRAN__ 95",
   "__GNUC__ 4",
   NULL
};
*/


      FILE *fp = fopen (macros, "r");
      int nlines = 0;
      int sizemax = 0;
      char buffer256[256];
      int i;
      int size, nline;
      int nread;

      if (fp == NULL)
	g95x_die ("Cannot open macros definition: `%s'\n", macros);

      while ((nread = fread (buffer256, 1, 256, fp)))
	{
	  size = 0;
	  for (i = 0; i < nread; i++)
	    {
	      if (buffer256[i] == '\n')
		{
		  nlines++;
		  if (size > sizemax)
		    sizemax = size;
		  size = 0;
		}
	      else
		{
		  size++;
		}
	    }
	}

      g95x_builtin_macros = g95_getmem (sizeof (char *) * nlines);

      rewind (fp);

      char *buffer = g95_getmem ((sizemax + 1) * sizeof (char));
      nline = 0;
      while ((nread = fread (buffer256, 1, 256, fp)))
	{
	  size = 0;
	  for (i = 0; i < nread; i++)
	    {
	      if (buffer256[i] == '\n')
		{
		  buffer[size++] = '\0';
		  g95x_builtin_macros[nline] =
		    g95_getmem (size * sizeof (char));
		  strcpy (g95x_builtin_macros[nline], buffer);
		  size = 0;
		  nline++;
		}
	      else
		{
		  buffer[size++] = buffer256[i];
		}
	    }
	}

      g95_free (buffer);

      fclose (fp);

      return 1;
    }

  if (get_opt_eq (&arch, argv[0], "xml-arch"))
    {
      FILE *fp = fopen (arch, "r");
      int iintr = 0, ilogl = 0, ireal = 0;


      if (fp == NULL)
	g95x_die ("Undefined arch : `%s'\n", arch);

      for (;;)
	{
	  int kind, radix, digits, bit_size;
	  fscanf (fp, "%d %d %d %d\n", &kind, &radix, &digits, &bit_size);
	  g95_integer_kinds[iintr].kind = kind;
	  g95_integer_kinds[iintr].radix = radix;
	  g95_integer_kinds[iintr].digits = digits;
	  g95_integer_kinds[iintr].bit_size = bit_size;
	  iintr++;
	  if (kind == 0)
	    break;
	}

      for (;;)
	{
	  int kind, bit_size;
	  fscanf (fp, "%d %d\n", &kind, &bit_size);
	  g95_logical_kinds[ilogl].kind = kind;
	  g95_logical_kinds[ilogl].bit_size = bit_size;
	  ilogl++;
	  if (kind == 0)
	    break;
	}

      for (;;)
	{
	  int kind, radix, totalsize, sign_start, exp_start,
	    exp_len, exp_bias, exp_nan, man_start, man_len;
	  char endian[32], msb[32];
	  fscanf (fp, "%d %d %s %d %d %d %d %d %d %d %d %s\n",
		  &kind, &radix, endian, &totalsize, &sign_start, &exp_start,
		  &exp_len, &exp_bias, &exp_nan, &man_start, &man_len, msb);
	  g95_real_kinds[ireal].kind = kind;
	  g95_real_kinds[ireal].radix = radix;
	  g95_real_kinds[ireal].endian = strcmp (endian, "END_LITTLE")
	    ? END_BIG : END_LITTLE;
	  g95_real_kinds[ireal].totalsize = totalsize;
	  g95_real_kinds[ireal].sign_start = sign_start;
	  g95_real_kinds[ireal].exp_start = exp_start;
	  g95_real_kinds[ireal].exp_len = exp_len;
	  g95_real_kinds[ireal].exp_bias = exp_bias;
	  g95_real_kinds[ireal].exp_nan = exp_nan;
	  g95_real_kinds[ireal].man_start = man_start;
	  g95_real_kinds[ireal].man_len = man_len;
	  g95_real_kinds[ireal].msb = strcmp (msb, "MSB_IMPLICIT")
	    ? MSB_EXPLICIT : MSB_IMPLICIT;
	  ireal++;
	  if (kind == 0)
	    break;
	}


      fclose (fp);
      return 1;
    }
  return -1;
}

int
g95x_init ()
{
  g95x_simplify_init ();
  return 1;
}

int
g95x_done ()
{
  return 1;
}






void
g95x_error (int die, int line, const char *file, const char *fmt, ...)
{
  va_list ap;

  fflush (stdout);
  if (g95x_out)
    fflush (g95x_out);

  fprintf (stderr, "\n");

  va_start (ap, fmt);
  vfprintf (stderr, fmt, ap);
  va_end (ap);

  fprintf (stderr, " at %s:%d\n", file, line);

  if (die)
    exit (1);
}


/*
 * g95x_valid_expr returns 1 if expr is a valid expr
 */


int
g95x_valid_expr (g95_expr * e)
{
  if (!e)
    return 0;
  if (!e->x.where.lb1)
    return 0;
  return 1;
}

try
g95x_simplify_expr (g95_expr ** expr)
{
  g95_expr *copy; 
  try t;
  g95_expr *expr1 = *expr;

  if (*expr == NULL)
    return SUCCESS;

  copy = g95_copy_expr (*expr);

  g95x_simplify_push (0);
  copy->x.full = expr1;
  t = g95_simplify_expr (copy);
  *expr = copy;
  g95x_simplify_pop ();



  return t;
}

try
g95x_simplify_spec_expr (g95_expr ** expr)
{
  g95_expr *copy = g95_copy_expr (*expr);
  try t;
  g95_expr *expr1 = *expr;

  if (*expr == NULL)
    return SUCCESS;

  g95x_simplify_push (0);
  t = g95_simplify_spec_expr (copy);

  *expr = copy;
  copy->x.full = expr1;

  g95x_simplify_pop ();

  return t;
}


/*
 * Locus functions; we use a static buffer as a FIFO to 
 * store matched locs of different types.
 *
 */

static g95x_ext_locus current_statement_locus[4096];	/* should be enough */

static int current_statement_locus_index = 0;

static int
compare_locus (g95x_locus * xw1, g95x_locus * xw2)
{
  if (xw1->lb1->linenum < xw2->lb1->linenum)
    return -1;
  if (xw1->lb1->linenum > xw2->lb1->linenum)
    return +1;
  if (xw1->column1 < xw2->column1)
    return -1;
  if (xw1->column1 > xw2->column1)
    return +1;
  return 0;
}

static void
g95x_push_locus0 (g95x_locus * loc, g95x_locus_type type,
		  g95_symbol * derived_type, g95_st_label * label,
		  g95_expr * e, g95_intrinsic_op op)
{
  if ((loc->lb1->linenum == loc->lb2->linenum)
      && (loc->column1 == loc->column2))
    return;

  while (current_statement_locus_index > 0)
    {
      if (compare_locus
	  (&current_statement_locus[current_statement_locus_index - 1].loc,
	   loc) < 0)
	break;
      current_statement_locus_index--;
    }

  memset (&current_statement_locus[current_statement_locus_index], 0,
	  sizeof (g95x_ext_locus));

  current_statement_locus[current_statement_locus_index].loc = *loc;
  current_statement_locus[current_statement_locus_index].type = type;

  current_statement_locus_index++;
}

void
g95x_push_locus (g95x_locus * loc, g95x_locus_type type)
{

  g95x_push_locus0 (loc, type, NULL, NULL, NULL, INTRINSIC_NONE);

}

void
g95x_push_locus_mmbr (g95x_locus * loc, g95_symbol * derived_type)
{

  g95x_push_locus0 (loc, G95X_LOCUS_MMBR, derived_type, NULL, NULL,
		    INTRINSIC_NONE);

}

void
g95x_push_locus_labl (g95x_locus * loc, g95_st_label * label)
{

  g95x_push_locus0 (loc, G95X_LOCUS_LABL, NULL, label, NULL, INTRINSIC_NONE);

}

void
g95x_push_locus_usop (g95x_locus * loc, g95_expr * e)
{

  g95x_push_locus0 (loc, G95X_LOCUS_NAME, NULL, NULL, e, INTRINSIC_NONE);

}


void
g95x_push_locus_inop (g95x_locus * loc, g95_intrinsic_op op)
{

  g95x_push_locus0 (loc, G95X_LOCUS_INOP, NULL, NULL, NULL, op);

}

g95x_locus
g95x_pop_locus ()
{
  static const g95x_locus xwn = { NULL, -1, NULL, -1 };
  if (current_statement_locus_index > 0)
    {
      current_statement_locus_index--;
      return current_statement_locus[current_statement_locus_index].loc;
    }
  return xwn;
}

g95x_ext_locus *
g95x_get_current_ext_locus_ref ()
{
  return current_statement_locus + (current_statement_locus_index - 1);
}

void
g95x_get_last_locus (g95x_locus * xw, g95x_locus_type type, int n)
{
  int i;

  xw->lb1 = xw->lb2 = NULL;
  xw->column1 = xw->column2 = 0;
  for (i = current_statement_locus_index - 1; i >= 0; i--)
    if (current_statement_locus[i].type & type)
      {
	if (n == 0)
	  {
	    *xw = current_statement_locus[i].loc;
	    return;
	  }
	else
	  {
	    n--;
	  }
      }
}

/*
 * Mark the beginning and end of locus
 */

void
g95x_mark_1 (g95x_locus * xw)
{
  xw->lb1 = g95_current_locus.lb;
  xw->column1 = g95_current_locus.column;
}

void
g95x_mark_2 (g95x_locus * xw)
{
  xw->lb2 = g95_current_locus.lb;
  xw->column2 = g95_current_locus.column;
}


void
g95x_mark_1_l (g95x_locus * xw, g95_locus * w)
{
  xw->lb1 = w->lb;
  xw->column1 = w->column;
}

void
g95x_mark_2_l (g95x_locus * xw, g95_locus * w)
{
  xw->lb2 = w->lb;
  xw->column2 = w->column;
}

int
g95x_locus_valid (g95x_locus * xw)
{
  return xw->lb1 && xw->lb2;
}

/* 
   Add a statement to the list of the current namespace
   g95x_push_statement does the job, and block_push,
   block_pop, block_switch are helper functions to manage
   blocks and branchs
 */

static int statement_block_index = 0;
static g95x_statement *statement_block[4096];	/* Should be enough */
static g95x_statement sta_program;

static g95x_statement *
block_push (g95x_statement * sta)
{
  statement_block[statement_block_index++] = sta;
  return sta;
}

static g95x_statement *
block_last ()
{
  if (statement_block_index == 0)
    return NULL;
  return statement_block[statement_block_index - 1];
}

static g95x_statement *
block_pop (g95x_statement * sta)
{
  g95x_statement *sta1 = NULL;

  if (statement_block_index > 0)
    sta1 = statement_block[--statement_block_index];
  else
    g95x_die ("Attempt to block_pop NULL\n");

  if (sta1)
    sta1->eblock = sta;

  /* for the IMPLIED_ENDDO we position the fblock temporarily 
   * it will be remove in set_statement_block_locus
   */
  if (sta1 != &sta_program)
    sta->fblock = sta1;

  return sta1;
}

static g95x_statement *
block_switch (g95x_statement * sta)
{
  if (statement_block_index > 0)
    {
      g95x_statement *sta1 = statement_block[statement_block_index - 1];
      statement_block[statement_block_index - 1] = sta;
      sta1->eblock = sta;
      if (sta1 != &sta_program)	/* Possible CONTAINS without PROGRAM */
	sta->fblock = sta1;
      return sta1;
    }
  else
    g95x_die ("Attempt to block_swich to NULL\n");
  return NULL;
}

static void
block_manage (g95x_statement * sta, g95_statement type)
{
  static int count = 0;


  if (count == 0)
    {
      sta_program.type = ST_PROGRAM;
      switch (type)
	{
	case ST_FUNCTION:
	case ST_SUBROUTINE:
	case ST_PROGRAM:
	case ST_MODULE:
	case ST_BLOCK_DATA:
	  break;
	default:
	  block_push (&sta_program);
	}
    }

  switch (type)
    {

    case ST_DO:
    case ST_IF_BLOCK:
    case ST_SELECT_CASE:
    case ST_WHERE_BLOCK:
    case ST_FORALL_BLOCK:
    case ST_SUBROUTINE:
    case ST_FUNCTION:
    case ST_INTERFACE:
    case ST_PROGRAM:
    case ST_MODULE:
    case ST_BLOCK_DATA:
    case ST_DERIVED_DECL:
    case ST_CRITICAL:
      block_push (sta);
      break;

    case ST_ELSEIF:
    case ST_ELSE:
    case ST_CASE:
    case ST_ELSEWHERE:
    case ST_CONTAINS:
    case ST_ENTRY:
      block_switch (sta);
      break;

    case ST_ENDDO:
    case ST_IMPLIED_ENDDO:
    case ST_ENDIF:
    case ST_END_SELECT:
    case ST_END_WHERE:
    case ST_END_TYPE:
    case ST_END_FORALL:
    case ST_END_SUBROUTINE:
    case ST_END_FUNCTION:
    case ST_END_INTERFACE:
    case ST_END_PROGRAM:
    case ST_END_MODULE:
    case ST_END_BLOCK_DATA:
    case ST_END_CRITICAL:
      block_pop (sta);
      break;
    default:
      break;
    }

  count++;
}


static g95x_statement *
get_statement ()
{
  return g95_getmem (sizeof (g95x_statement));
}

void
g95x_free_rename_list (g95_use_rename * list)
{
  g95_use_rename *next;

  for (; list; list = next)
    {
      next = list->next;
      g95_free (list);
    }
}

static void
free_implicit_list (g95x_implicit_spec * is)
{
  g95x_implicit_spec *is1;
  for (is1 = is; is1; is1 = is)
    {
      is = is1->next;
      g95_free (is1);
    }
}

static void
free_statement (g95x_statement * sta)
{
  /* we should take care of allocated interface structures */
  if (sta->type == ST_IMPORT)
    g95x_free_list (&sta->u.import.decl_list);
  if (sta->type == ST_MODULE_PROC)
    g95x_free_list (&sta->u.modproc.decl_list);
  if (sta->type == ST_ATTR_DECL)
    g95x_free_list (&sta->u.data_decl.decl_list);
  if (sta->type == ST_DATA_DECL)
    {
      g95x_free_list (&sta->u.data_decl.decl_list);
      if (sta->u.data_decl.as)
	g95_free_array_spec ((g95_array_spec *) sta->u.data_decl.as);
      if (sta->u.data_decl.cas)
	g95_free_coarray_spec ((g95_coarray_spec *) sta->u.data_decl.cas);
    }
  if (sta->type == ST_NAMELIST)
    g95x_free_list (&sta->u.namelist.namelist_list);
  if (sta->type == ST_COMMON)
    g95x_free_list (&sta->u.common.common_list);
  if (sta->type == ST_ENUMERATOR)
    g95x_free_list (&sta->u.enumerator.enumerator_list);
  if (sta->type == ST_PARAMETER)
    g95x_free_list (&sta->u.data_decl.decl_list);
  if (sta->type == ST_USE)
    g95x_free_rename_list (sta->u.use.rename_list);
  if (sta->type == ST_IMPLICIT)
    free_implicit_list (sta->u.implicit.list_is);
  if (sta->ext_locs)
    g95_free (sta->ext_locs);
  g95_free (sta);
}

void
g95x_free_namespace (g95_namespace * ns)
{
  g95x_statement *sta;
  g95x_kind *xkind, *xkind1;
  g95x_statement *sta1;
  for (sta = ns->x.statement_head; sta; sta = sta1)
    {
      sta1 = sta->next;
      free_statement (sta);
    }


  for (xkind = ns->x.k_list; xkind; xkind = xkind1)
    {
      g95_free_expr (xkind->kind);
      xkind1 = xkind->next;
      g95_free (xkind);
    }
  ns->x.k_list = NULL;
  ns->x.statement_head = ns->x.statement_tail = NULL;
}

static int
compare_locus_edge (g95x_locus * loc1, g95x_locus * loc2, int k)
{
  int l1, c1, l2, c2;
  switch (k)
    {
    case 1:
      l1 = loc1->lb1->linenum;
      c1 = loc1->column1;
      l2 = loc2->lb1->linenum;
      c2 = loc2->column1;
      break;
    case 2:
      l1 = loc1->lb2->linenum;
      c1 = loc1->column2;
      l2 = loc2->lb2->linenum;
      c2 = loc2->column2;
      break;
    default:
      g95x_die ("Unexpected position in compare_locus_edge\n");
    }

  if (l1 < l2)
    return -1;
  if (l1 > l2)
    return +1;
  if (c1 < c2)
    return -1;
  if (c1 > c2)
    return +1;

  return 0;
}

static g95x_statement *
get_statement_with_locs (g95x_locus * where)
{
  g95x_statement *sta;
  /* Set locs in current statement */
  int i, k;
  int clab = 0;

  for (i = 0, k = 0; i < current_statement_locus_index; i++)
    {
      if ((compare_locus_edge (&current_statement_locus[i].loc, where, 1) >=
	   0)
	  && (compare_locus_edge (&current_statement_locus[i].loc, where, 2)
	      <= 0))
	if (current_statement_locus[i].type != G95X_LOCUS_LANG)
	  k++;
      /* kludge for labels */
      if ((i == 0) && (current_statement_locus[i].type == G95X_LOCUS_LABL))
	{
	  int j;
	  int j0 = current_statement_locus[i].loc.column2 - 1;
	  int j1 = where->column1 - 1;
	  g95_linebuf *lb1 = current_statement_locus[i].loc.lb1;
	  g95_linebuf *lb2 = where->lb1;
	  clab = 1;
	  if (lb1 == lb2)
	    for (j = j0; j < j1; j++)
	      clab = clab && (lb1->mask[j] == ' ');
	  else
	    clab = 0;
	  if (clab)
	    k++;
	}
    }

  sta = get_statement ();
  sta->where = *where;

  sta->n_ext_locs = k;
  sta->ext_locs = g95_getmem (k * sizeof (g95x_ext_locus));

  int iel = 0;

  for (i = 0; i < current_statement_locus_index; i++)
    {
      if ((compare_locus_edge (&current_statement_locus[i].loc, where, 1) >=
	   0)
	  && (compare_locus_edge (&current_statement_locus[i].loc, where, 2)
	      <= 0))
	if (current_statement_locus[i].type != G95X_LOCUS_LANG)
	  sta->ext_locs[iel++] = current_statement_locus[i];
      if ((i == 0) && clab)
	sta->ext_locs[iel++] = current_statement_locus[i];
    }

  return sta;
}







/*
 * Set the right code for the statement; that's because io statements
 * span several code
 */

static void
set_statement_code (g95x_statement * sta, g95_code * c)
{


  switch (sta->type)
    {
    case ST_NULLIFY:
      {
	g95_code *c2 = c;
	int n = 1;
	for (; c2->next; c2 = c2->next, n++);
	sta->u.nullify.n = n;
      }
      break;
    default:
      break;
    }

  sta->code = c;
  if (c)
    c->x.statement = sta;

}


static void
manage_statement_list (g95x_statement ** statement_head,
		       g95x_statement ** statement_tail,
		       g95x_statement ** sta, int ns)
{
  if (ns)
    {
      if (*statement_tail)
	{
	  (*statement_tail)->next = *sta;
	  (*sta)->prev = (*statement_tail);
	  (*statement_tail) = *sta;
	}
      else
	{
	  (*statement_head) = (*statement_tail) = *sta;
	}
    }
  else
    {
      if (*statement_tail)
	{
	  (*statement_tail)->a_next = *sta;
	  (*sta)->a_prev = (*statement_tail);
	  (*statement_tail) = *sta;
	}
      else
	{
	  (*statement_head) = (*statement_tail) = *sta;
	}
    }
}


#define case_ns_block1 \
  case ST_SUBROUTINE: case ST_FUNCTION: case ST_PROGRAM: \
  case ST_BLOCK_DATA: case ST_MODULE

#define case_ex_block1 \
  case ST_DERIVED_DECL: case ST_IF_BLOCK: case ST_DO:         \
  case ST_SELECT_CASE: case ST_WHERE_BLOCK: case ST_CRITICAL: \
case ST_FORALL_BLOCK: case ST_INTERFACE

#define case_block1 \
  case_ns_block1: case_ex_block1

#define case_block2 \
  case ST_ELSEIF: case ST_ELSE: case ST_CASE: case ST_ELSEWHERE: case ST_CONTAINS

#define case_ns_block3 \
  case ST_END_SUBROUTINE: case ST_END_FUNCTION: case ST_END_PROGRAM:  \
  case ST_END_BLOCK_DATA: case ST_END_MODULE

#define case_ex_block3 \
  case ST_END_TYPE: case ST_ENDIF: case ST_ENDDO:               \
  case ST_END_SELECT: case ST_END_WHERE: case ST_END_CRITICAL:  \
  case ST_END_FORALL: case ST_END_INTERFACE

#define case_block3 \
  case_ns_block3: case_ex_block3

static void
set_statement_block_locus (g95x_statement * sta)
{
  int type = sta->type;

  switch (type)
    {
    case_ex_block3:
      {
	g95x_statement *sta1 = sta, *sta2 = sta;
	while (1)
	  {
	    if (sta1->fblock)
	      sta1 = sta1->fblock;
	    else
	      break;
	  }
	sta1->where_construct.lb1 = sta1->where.lb1;
	sta1->where_construct.column1 = sta1->where.column1;
	sta1->where_construct.lb2 = sta2->where.lb2;
	sta1->where_construct.column2 = sta2->where.column2;
      }
    default:
      break;
    }

  switch (type)
    {
    case_ns_block3:
      {
	g95_namespace *ns = sta->ns;
	g95x_statement *head = sta->ns->x.statement_head,
	  *tail = sta->ns->x.statement_tail;
	ns->x.where.lb1 = head->where.lb1;
	ns->x.where.column1 = head->where.column1;
	ns->x.where.lb2 = tail->where.lb2;
	ns->x.where.column2 = tail->where.column2;
      }
    default:
      break;
    }

  switch (type)
    {
    case_ex_block3:
      {
	g95x_statement *sta1 = sta, *sta2 = sta;
	while (1)
	  {
	    if (sta1->fblock)
	      sta1 = sta1->fblock;
	    else
	      break;
	  }
	sta1->where_construct.lb1 = sta1->where.lb1;
	sta1->where_construct.column1 = sta1->where.column1;
	sta1->where_construct.lb2 = sta2->where.lb2;
	sta1->where_construct.column2 = sta2->where.column2;
      }
    default:
      break;
    }

  switch (type)
    {
    case_ns_block3:
      {
	g95_namespace *ns = sta->ns;
	g95x_statement *head = sta->ns->x.statement_head,
	  *tail = sta->ns->x.statement_tail;
	ns->x.where.lb1 = head->where.lb1;
	ns->x.where.column1 = head->where.column1;
	ns->x.where.lb2 = tail->where.lb2;
	ns->x.where.column2 = tail->where.column2;
      }
    default:
      break;
    }

  if (sta->implied_enddo)
    goto block3b;

  switch (type)
    {
    case_block2:
    case_ex_block3:
    case_ns_block3:
    block3b:
      {
	g95x_statement * sta1 = sta->fblock, *sta2 = sta->a_prev;
	if (sta1)
	  {			/* CONTAINS/END_PROGRAM without PROGRAM */
            g95x_statement * sta3 = sta1;
	    sta1->where_block.lb1 = sta3->where.lb1;
	    sta1->where_block.column1 = sta3->where.column1;
	    sta1->where_block.lb2 = sta2->where.lb2;
	    sta1->where_block.column2 = sta2->where.column2;
	  }
	else if ((sta->type != ST_END_PROGRAM) && (sta->type != ST_CONTAINS))
	  {
	    g95x_die ("");
	  }
      }
    default:
      break;
    }

  if (sta->implied_enddo)
    {
      g95x_statement *sta1 = sta->fblock;
      sta1->where_construct = sta1->where_block;
      sta->fblock = NULL;
    }


}


static void
set_statement_block (g95x_statement * sta)
{
  int type = sta->type;

  switch (type)
    {
    case_block1:
      sta->block = g95_new_block;
    default:
      break;
    }

  switch (type)
    {
    case_block2:
      {
	g95x_statement *sta1 = block_last ();	/* CONTAINS without PROGRAM */
	if (sta1)
	  sta->block = sta1->block;
      }
    default:
      break;
    }

  if (type == ST_ENTRY)
    sta->block = sta->code->sym;

  if (type == ST_STATEMENT_FUNCTION)
    sta->block = g95x_stmt_func_take_symbol ();

  switch (type)
    {
    case_block3:
      {
	g95x_statement *sta1 = block_last ();	/* END_PROGRAM without PROGRAM */
	if (sta1)
	  sta->block = sta1->block;
      }
    default:
      break;
    }

}

static void
fixup_ST_SUBROUTINE (g95x_statement * sta, g95_namespace * ns)
{
  g95x_statement *sta1 = block_last ();
  if (sta1 && (sta1->type == ST_INTERFACE))
    {
      g95x_symbol_list *slist = g95x_get_symbol_list ();
      slist->symbol = g95_state_stack->sym;
      if (sta1->u.interface.symbol_tail)
	{
	  sta1->u.interface.symbol_tail->next = slist;
	  sta1->u.interface.symbol_tail = slist;
	}
      else
	{
	  sta1->u.interface.symbol_head =
	    sta1->u.interface.symbol_tail = slist;
	}
    }
  sta->u.subroutine.attr = *(g95x_get_current_attr ());
  g95x_take_bind ();
}

static void
fixup_ST_FUNCTION (g95x_statement * sta, g95_namespace * ns)
{
  fixup_ST_SUBROUTINE (sta, ns);
  sta->u.function.ts = *(g95x_get_current_ts ());
  g95x_take_bind ();
}

static void
fixup_ST_INTERFACE (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.interface.info.type = current_interface.type;
  sta->u.interface.info.ns = (struct g95_namespace *) current_interface.ns;
  sta->u.interface.info.uop = (struct g95_user_op *) current_interface.uop;
  sta->u.interface.info.generic =
    (struct g95_symtree *) current_interface.generic;
  sta->u.interface.info.op = current_interface.op;
}

static void
fixup_ST_END_INTERFACE (g95x_statement * sta, g95_namespace * ns)
{
  g95x_statement *sta1 = block_last ();
  sta->u.interface = sta1->u.interface;
  sta->u.interface.st_interface = sta1;
}

static void
fixup_ST_FORMAT (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.format.here = new_st.here;
}

static void
fixup_ST_EQUIVALENCE (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.equiv.e1 = ns->equiv;
  sta->u.equiv.e2 = ns->x.equiv1;
  ns->x.equiv1 = ns->equiv;
}

static void
fixup_ST_DATA (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.data.d1 = ns->data;
  sta->u.data.d2 = ns->x.data1;
  ns->x.data1 = ns->data;
}


static void
fixup_ST_USE (g95x_statement * sta, g95_namespace * ns)
{
  ns->x.equiv1 = ns->equiv;
  sta->u.use.only = g95x_get_only_flag ();
  sta->u.use.rename_list = g95x_module_take_rename_list ();
  sta->u.use.module = g95x_module_get_mod ();
}


static void
fixup_ST_IMPLICIT (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.implicit.list_is = g95x_take_implicit_spec_list ();
}


static void
fixup_ST_ENUM (g95x_statement * sta, g95_namespace * ns)
{
  sta->enum_bindc = g95x_enum_bindc ();
}

static void
fixup_ST_DATA_DECL (g95x_statement * sta, g95_namespace * ns)
{
  g95x_statement *sta1 = block_last ();
  g95x_voidp_list *sl;
  if (sta1)
    sta->in_derived = sta1->type == ST_DERIVED_DECL;
  sta->u.data_decl.ts = *(g95x_get_current_ts ());
  sta->u.data_decl.as = (struct g95_array_spec *) g95x_take_current_as ();
  sta->u.data_decl.cas = (struct g95_coarray_spec *) g95x_take_current_cas ();
  sta->u.data_decl.attr = *(g95x_get_current_attr ());
  sta->u.data_decl.decl_list = g95x_decl_take_list ();
  sta->u.data_decl.bind = g95x_take_bind ();
  for (sl = sta->u.data_decl.decl_list; sl; sl = sl->next)
    sl->in_derived = sta->in_derived;
}

static void
fixup_ST_DERIVED_DECL (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.data_decl.attr = *(g95x_get_current_attr ());
}

static void
fixup_ST_PARAMETER (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.data_decl.decl_list = g95x_decl_take_list ();
}

static void
fixup_ST_ATTR_DECL (g95x_statement * sta, g95_namespace * ns)
{

  sta->u.data_decl.decl_list = g95x_decl_take_list ();
  sta->u.data_decl.attr = *(g95x_get_current_attr ());
  sta->u.data_decl.bind = g95x_take_bind ();

}

static void
fixup_ST_MODULE_PROC (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.modproc.decl_list = g95x_decl_take_list ();
}

static void
fixup_ST_IMPORT (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.import.decl_list = g95x_import_take_list ();
}

static void
fixup_ST_COMMON (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.common.common_list = g95x_common_take_list ();
}

static void
fixup_ST_ENUMERATOR (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.enumerator.enumerator_list = g95x_enumerator_take_list ();
}

static void
fixup_ST_NAMELIST (g95x_statement * sta, g95_namespace * ns)
{
  sta->u.namelist.namelist_list = g95x_naml_take_list ();
}


/*
 * Build and push a statement once it is matched. 
 */

g95x_statement *g95x_statement_head = NULL, *g95x_statement_tail = NULL;

void
g95x_push_statement (g95_statement type, g95_code * code, g95x_locus * where)
{
  g95x_statement *sta;
  g95_namespace *ns;


  ns = (type == ST_MODULE_PROC) ? g95_current_ns->parent : g95_current_ns;


  if (type == ST_IMPLIED_ENDDO)
    {
      sta = ns->x.statement_tail;
      sta->implied_enddo++;
      goto block;
    }

  sta = get_statement_with_locs (where);

  sta->type = type;

  sta->ns = ns;

  sta->here = g95_statement_label;

  if (sta->here)
    {
      sta->where.lb1 = sta->here->x.where.lb1;
      sta->where.column1 = sta->here->x.where.column1;
    }

  set_statement_code (sta, code);

  manage_statement_list (&ns->x.statement_head, &ns->x.statement_tail, &sta,
			 1);
  manage_statement_list (&g95x_statement_head, &g95x_statement_tail, &sta, 0);


  set_statement_block (sta);

  switch (type)
    {
    case ST_IMPORT:
      fixup_ST_IMPORT (sta, ns);
      break;
    case ST_MODULE_PROC:
      fixup_ST_MODULE_PROC (sta, ns);
      break;
    case ST_NAMELIST:
      fixup_ST_NAMELIST (sta, ns);
      break;
    case ST_SUBROUTINE:
      fixup_ST_SUBROUTINE (sta, ns);
      break;
    case ST_FUNCTION:
      fixup_ST_FUNCTION (sta, ns);
      break;
    case ST_INTERFACE:
      fixup_ST_INTERFACE (sta, ns);
      break;
    case ST_END_INTERFACE:
      fixup_ST_END_INTERFACE (sta, ns);
      break;
    case ST_FORMAT:
      fixup_ST_FORMAT (sta, ns);
      break;
    case ST_EQUIVALENCE:
      fixup_ST_EQUIVALENCE (sta, ns);
      break;
    case ST_DATA:
      fixup_ST_DATA (sta, ns);
      break;
    case ST_USE:
      fixup_ST_USE (sta, ns);
      break;
    case ST_IMPLICIT:
      fixup_ST_IMPLICIT (sta, ns);
      break;
    case ST_ENUM:
      fixup_ST_ENUM (sta, ns);
      break;
    case ST_DATA_DECL:
      fixup_ST_DATA_DECL (sta, ns);
      break;
    case ST_DERIVED_DECL:
      fixup_ST_DERIVED_DECL (sta, ns);
      break;
    case ST_PARAMETER:
      fixup_ST_PARAMETER (sta, ns);
      break;
    case ST_ATTR_DECL:
      fixup_ST_ATTR_DECL (sta, ns);
      break;
    case ST_COMMON:
      fixup_ST_COMMON (sta, ns);
      break;
    case ST_ENUMERATOR:
      fixup_ST_ENUMERATOR (sta, ns);
      break;
    default:
      break;
    }



block:
  block_manage (sta, type);

  set_statement_block_locus (sta);

  switch (type)
    {
      g95_code *c = NULL;
    case ST_FORALL:
    case ST_SIMPLE_IF:
      c = sta->code->block;
      goto push;
    case ST_WHERE:
      c = sta->code->block->next;

    push:
      g95_statement_label = NULL;

      g95x_push_statement (c->x.sta, c, &c->x.where);

      g95_statement_label = sta->here;

      break;
    default:
      break;
    }


  current_statement_locus_index = 0;

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


static void
_fixup_symbols (g95_symtree * st, void *data)
{
  g95_namespace *ns = data;
  g95_symbol *s = st->n.sym;


  if (st->ambiguous)
    return;

  s->x.is_generic =
    g95_generic_sym (st->name, ns, s->ts.type == BT_UNKNOWN ? 1 : 0);


  if ((s->module && (!strcmp (s->module, "(intrinsic)")))
      || s->attr.intrinsic)
    {
      g95_intrinsic_sym *is;
      is =
	s->ts.type !=
	BT_PROCEDURE ? g95_find_function (s->name) : g95_find_subroutine (s->
									  name);
      g95x_mark_intrinsic_used (is);
      s->x.is_generic = 0;
    }

  if (!strcmp (st->name, s->name))
    s->x.visible++;

  return;
}



static void
traverse_symtree (g95_symtree * st, void (*func) (g95_symtree *, void *),
		  void *data)
{

  if (st != NULL)
    {

      traverse_symtree (st->left, func, data);
      (*func) (st, data);
      traverse_symtree (st->right, func, data);

    }
}

static void
fixup_symbols (g95_namespace * ns)
{

  if (ns->sym_root == NULL)
    return;

  traverse_symtree (ns->sym_root, _fixup_symbols, ns);

}

static void
_clear_generic (g95_symtree * st, void *data)
{
  g95_interface *intf = st->n.generic;
  for (; intf; intf = intf->next)
    intf->sym->x.is_generic = 0;
}


static void
clear_generics (g95_namespace * ns)
{

  if (ns->generic_root == NULL)
    return;

  traverse_symtree (ns->generic_root, _clear_generic, NULL);

}


void
fixup_generics (g95_namespace * ns)
{
  g95_namespace *ns1;
  g95x_statement *sta;

  if (!ns)
    return;

  for (ns1 = ns; ns1; ns1 = ns1->sibling)
    {

      for (sta = ns1->x.statement_head; sta; sta = sta->next)
	{
	  if (sta->type == ST_INTERFACE)
	    {
	      g95x_symbol_list *sl;
	      for (sl = sta->u.interface.symbol_head; sl; sl = sl->next)
		fixup_generics (sl->symbol->formal_ns);
	    }
	  if (sta == ns1->x.statement_tail)
	    break;
	}

      fixup_symbols (ns1);
      clear_generics (ns1);

      fixup_generics (ns1->contained);
    }
}

void
g95x_fixup_generics (g95_namespace * ns)
{

  fixup_generics (ns);


}







void
g95x_push_list_symbol (g95x_voidp_list ** list, g95_symbol * s)
{
  g95x_voidp_list *sl;
  if (!g95x_option.enable)
    return;
  sl = g95x_get_voidp_list ();
  sl->u.symbol = s;
  sl->type = G95X_VOIDP_SYMBOL;
  sl->next = *list;
  *list = sl;
  g95x_get_last_locus (&sl->where, G95X_LOCUS_NAME, 0);
}

void
g95x_push_list_common (g95x_voidp_list ** list, g95_common_head * c)
{
  g95x_voidp_list *sl;
  if (!g95x_option.enable)
    return;
  sl = g95x_get_voidp_list ();
  sl->u.common = (struct g95_common_head *) c;
  sl->type = G95X_VOIDP_COMMON;
  sl->next = *list;
  *list = sl;
  g95x_get_last_locus (&sl->where, G95X_LOCUS_NAME, 0);
}

void
g95x_set_dimension (g95x_voidp_list ** list, void *s)
{
  if (!g95x_option.enable)
    return;
  g95x_voidp_list *sl = *list;
  for (; sl; sl = sl->next)
    {
      if ((sl->type == G95X_VOIDP_SYMBOL) || (sl->type == G95X_VOIDP_COMPNT))
	if (sl->u.voidp == s)
	  {
	    sl->dimension = 1;
	    return;
	  }
    }
  g95x_die ("Cannot find symbol %R in common_list");
}

void
g95x_free_list (g95x_voidp_list ** list)
{
  g95x_voidp_list *sl, *sl1;
  if (!g95x_option.enable)
    return;
  for (sl = *list; sl;)
    {
      sl1 = sl->next;
      g95_free (sl);
      sl = sl1;
    }
  *list = NULL;
}

g95x_voidp_list *
g95x_take_list (g95x_voidp_list ** list)
{
  g95x_voidp_list *sl = *list;
  *list = NULL;
  return sl;
}

void
g95x_set_length (g95x_voidp_list ** list, void *s)
{
  g95x_voidp_list *sl;
  if (!g95x_option.enable)
    return;
  sl = *list;
  for (; sl; sl = sl->next)
    {
      if ((sl->type == G95X_VOIDP_SYMBOL) || (sl->type == G95X_VOIDP_COMPNT))
	if (sl->u.voidp == s)
	  {
	    sl->length = 1;
	    return;
	  }
    }
  g95x_die ("Cannot find symbol %R in decl_list");
}

void
g95x_set_initialization (g95x_voidp_list ** list, g95_symbol * s)
{
  g95x_voidp_list *sl;
  if (!g95x_option.enable)
    return;
  sl = *list;
  for (; sl; sl = sl->next)
    {
      if (sl->type == G95X_VOIDP_SYMBOL)
	if (sl->u.symbol == s)
	  {
	    sl->init = 1;
	    return;
	  }
    }
  g95x_die ("Cannot find symbol %R in decl_list");
}

void
g95x_push_list_generic (g95x_voidp_list ** list, g95_symtree * s)
{
  g95x_voidp_list *sl;
  if (!g95x_option.enable)
    return;
  sl = g95x_get_voidp_list ();
  sl->u.generic = s;
  sl->type = G95X_VOIDP_GENERIC;
  sl->next = *list;
  *list = sl;
  g95x_get_last_locus (&sl->where, G95X_LOCUS_NAME, 0);
}

void
g95x_push_list_user_op (g95x_voidp_list ** list, g95_symtree * s)
{
  g95x_voidp_list *sl;
  if (!g95x_option.enable)
    return;
  sl = g95x_get_voidp_list ();
  sl->u.userop = s;
  sl->type = G95X_VOIDP_USEROP;
  sl->next = *list;
  *list = sl;
  g95x_get_last_locus (&sl->where, G95X_LOCUS_NAME, 0);
}

void
g95x_push_list_intr_op_idx (g95x_voidp_list ** list, int s)
{
  g95x_voidp_list *sl;
  if (!g95x_option.enable)
    return;
  sl = g95x_get_voidp_list ();
  sl->u.introp_idx = s;
  sl->type = G95X_VOIDP_INTROP_IDX;
  sl->next = *list;
  *list = sl;
  g95x_get_last_locus (&sl->where, G95X_LOCUS_INOP, 0);
}

void
g95x_push_list_component (g95x_voidp_list ** list, g95_component * s)
{
  g95x_voidp_list *sl;
  if (!g95x_option.enable)
    return;
  sl = g95x_get_voidp_list ();
  sl->u.component = s;
  sl->type = G95X_VOIDP_COMPNT;
  sl->next = *list;
  *list = sl;
  g95x_get_last_locus (&sl->where, G95X_LOCUS_NAME, 0);
}





const void *
g95x_resolve_intrinsic_operator (g95_intrinsic_op type)
{
  g95_namespace *ns = g95x_dump_ns;
  while (ns)
    {
      if (ns->operator[type])
	return &ns->operator[type];
      if (ns->interface && (!ns->import))
	break;
      ns = ns->parent;
    }
  return g95x_iop_type (type);
}

const void *
g95x_resolve_generic (const char *name)
{
  void *p = NULL;
  g95_intrinsic_sym *isym;
  g95_namespace *ns = g95x_dump_ns;
  while (ns)
    {
      p = g95_find_generic ((char *) name, ns);
      if (p)
	return p;
      if (ns->interface && (!ns->import))
	break;
      ns = ns->parent;
    }
  isym = g95_find_function ((char *) name);
  if (isym && isym->x.used)
    return &isym->generic;
  isym = g95_find_subroutine ((char *) name);
  if (isym && isym->x.used)
    return &isym->generic;
  g95x_die ("");
  return NULL;
}

const void *
g95x_resolve_user_operator (const char *name)
{
  void *p = NULL;
  g95_namespace *ns = g95x_dump_ns;
  while (ns)
    {
      p = g95_find_uop ((char *) name, ns);
      if (p)
	return p;
      if (ns->interface && (!ns->import))
	break;
      ns = ns->parent;
    }
  g95x_die ("");
  return NULL;
}
