#include "xml-dump.h"



#ifndef INTF

static int
compare_item_locus (const void *ila, const void *ilb)
{
  g95x_item_locus *const *_ila = ila, *const *_ilb = ilb;
  return g95x_compare_locus ((*_ila)->xwl, (*_ilb)->xwl);
}

static void
dump_lst (g95x_item_locus * lst, int bn)
{
  int i, n;
  g95x_item_locus *il;
  g95x_item_locus **tab;
  for (il = lst, n = 0; il; il = il->next, n++);
  tab = (g95x_item_locus **) g95_getmem (sizeof (g95x_item_locus *) * n);
  for (il = lst, n = 0; il; il = il->next, n++)
    tab[n] = il;
  qsort (tab, n, sizeof (g95x_item_locus *), compare_item_locus);

  for (i = 0; i < n; i++)
    {
      il = tab[i];
      G95X_DUMP_SRCA (il->xwl);
      if (bn)
	G95X_ADD_SCT (il->key);
      il->dmp (il->obj);
      if (bn)
	G95X_END_SCT ();
      g95_free (il);
    }

  g95_free (tab);

}

static void
dump_lst_na (g95x_item_locus * lst)
{
  dump_lst (lst, 1);
}

static void
dump_lst_an (g95x_item_locus * lst)
{
  dump_lst (lst, 0);
}

#endif


#ifndef INTF


static void
dump_global_symbol_ref_na (const char *key, g95_symbol * sym, g95x_locus * xw,
			   int global)
{
  g95x_symbol_ref sr;

  if (global)
    {
      strcpy (sr.als, g95x_symbol_gname (sym));
      sr.cls = _S(G95X_S_SYMBOL_GREF);
    }
  else
    {
      g95x_get_name_substr (xw, sr.als);

      if (sr.als[0] == '\0')
	g95x_die ("");

      sr.strcmp = strcmp (sym->name, sr.als);
      sr.cls = _S(G95X_S_SYMBOL_REF);
    }

  sr.sym = sym;
  sr.xw = xw;
  sr.type = NULL;

  G95X_DUMP_SUBOBJ_NA (key, symbol_ref, &sr);
}


static void
dump_global_symbol_ref_an (g95_symbol * sym, g95x_locus * xw)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_SYMBOL_REF);
  sr.strcmp = strcmp (sym->name, sr.als);
  sr.type = NULL;

  G95X_DUMP_SUBOBJ_AN (symbol_ref, &sr);
}

static void
dump_local_symbol_ref_na (const char *key, g95_symbol * sym, g95x_locus * xw)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_SYMBOL_REF);
  sr.strcmp = strcmp (sym->name, sr.als);
  sr.type = NULL;

  G95X_DUMP_SUBOBJ_NA (key, symbol_ref, &sr);
}

static void
dump_local_symbol_ref_an (g95_symbol * sym, g95x_locus * xw)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_SYMBOL_REF);
  sr.strcmp = strcmp (sym->name, sr.als);
  sr.type = NULL;

  G95X_DUMP_SUBOBJ_AN (symbol_ref, &sr);
}

static const char* component_type_name (const g95_component* c,
                                        g95_typespec *ts)  
{
  const char* type;
  static char type1[G95_MAX_SYMBOL_LEN+1];
  g95_symbol *sym = c->x.derived;

  if (ts->x.where_derived.lb1) {
    g95x_get_name_substr (&ts->x.where_derived, type1);
    return type1;
  }

  if (g95x_symbol_global (sym) && !g95x_symbol_defined (sym))
    {
      type = g95x_symbol_gname (sym);
    }
  else
    {
      type = sym->name;
    }
  return type;
}

static void
dump_local_component_ref_na (const char *key, const g95_component *c,
			     g95x_locus * xw, g95_typespec* ts)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = c;
  sr.xw = xw;
  sr.cls = _S(G95X_S_COMPONENT_REF);
  sr.strcmp = 0;
  sr.type = component_type_name (c, ts);

  G95X_DUMP_SUBOBJ_NA (key, symbol_ref, &sr);
}

static void
dump_local_component_ref_an (const g95_component *c, g95x_locus * xw, 
                             g95_typespec* ts)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = c;
  sr.xw = xw;
  sr.cls = _S(G95X_S_COMPONENT_REF);
  sr.strcmp = 0;
  sr.type = component_type_name (c, ts);

  G95X_DUMP_SUBOBJ_AN (symbol_ref, &sr);
}

static void
dump_local_common_ref_na (const char *key, const void *sym, g95x_locus * xw)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_COMMON_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  G95X_DUMP_SUBOBJ_NA (key, symbol_ref, &sr);
}

static void
dump_local_common_ref_an (const void *sym, g95x_locus * xw)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_COMMON_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  G95X_DUMP_SUBOBJ_AN (symbol_ref, &sr);
}

static void
dump_local_generic_ref_na (const char *key, const void *sym, g95x_locus * xw)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_GENERIC_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  G95X_DUMP_SUBOBJ_NA (key, symbol_ref, &sr);
}

static void
dump_local_generic_ref_an (const void *sym, g95x_locus * xw)
{
  g95x_symbol_ref sr;

  g95x_get_name_substr (xw, sr.als);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_GENERIC_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  G95X_DUMP_SUBOBJ_AN (symbol_ref, &sr);
}

static void
dump_local_user_operator_ref_na (const char *key, const void *sym,
				 g95x_locus * xw, int opl)
{
  g95x_symbol_ref sr;

  g95x_get_code_substr (xw, sr.als, 2 * G95_MAX_SYMBOL_LEN);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_OPERATOR_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  if (opl)
    {
      g95x_user_operator1_ref uor;
      uor.sr = sr;
      g95x_opl (xw, &uor.xwo, "operator(", ")");
      G95X_DUMP_SUBOBJ_NA (key, user_operator1_ref, &uor);
    }
  else
    {
      G95X_DUMP_SUBOBJ_NA (key, symbol_ref, &sr);
    }
}

static void
dump_local_user_operator_ref_an (const void *sym, g95x_locus * xw, int opl)
{
  g95x_symbol_ref sr;

  g95x_get_code_substr (xw, sr.als, 2 * G95_MAX_SYMBOL_LEN);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_OPERATOR_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  if (opl)
    {
      g95x_user_operator1_ref uor;
      uor.sr = sr;
      g95x_opl (xw, &uor.xwo, "operator(", ")");
      G95X_DUMP_SUBOBJ_AN (user_operator1_ref, &uor);
    }
  else
    {
      G95X_DUMP_SUBOBJ_AN (symbol_ref, &sr);
    }
}

static void
dump_local_intrinsic_operator_ref_na (const char *key, const void *sym,
				      g95x_locus * xw, int opl)
{
  g95x_symbol_ref sr;
  char tmp[2 * G95_MAX_SYMBOL_LEN + 1];

  g95x_get_code_substr (xw, tmp, 2 * G95_MAX_SYMBOL_LEN);

  if (tmp[0] == '\0')
    g95x_die ("");

  g95x_xml_escape (sr.als, tmp, 2 * G95_MAX_SYMBOL_LEN);

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_OPERATOR_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  if (opl)
    {
      g95x_intrinsic_operator1_ref uor;
      uor.sr = sr;
      g95x_opl (xw, &uor.xwo, "operator(", ")");
      G95X_DUMP_SUBOBJ_NA (key, intrinsic_operator1_ref, &uor);
    }
  else
    {
      G95X_DUMP_SUBOBJ_NA (key, symbol_ref, &sr);
    }
}

static void
dump_local_intrinsic_operator_ref_an (const void *sym, g95x_locus * xw,
				      int opl)
{
  g95x_symbol_ref sr;

  g95x_get_code_substr (xw, sr.als, 2 * G95_MAX_SYMBOL_LEN);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_OPERATOR_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  if (opl)
    {
      g95x_intrinsic_operator1_ref uor;
      uor.sr = sr;
      g95x_opl (xw, &uor.xwo, "operator(", ")");
      G95X_DUMP_SUBOBJ_AN (intrinsic_operator1_ref, &uor);
    }
  else
    {
      G95X_DUMP_SUBOBJ_AN (symbol_ref, &sr);
    }
}

static void
dump_local_assignment_ref_na (const char *key, const void *sym,
			      g95x_locus * xw, int opl)
{
  g95x_symbol_ref sr;

  g95x_get_code_substr (xw, sr.als, 2 * G95_MAX_SYMBOL_LEN);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_ASSIGNMENT_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  if (opl)
    {
      g95x_assignment1_ref uor;
      uor.sr = sr;
      g95x_opl (xw, &uor.xwo, "assignment(", ")");
      G95X_DUMP_SUBOBJ_NA (key, assignment1_ref, &uor);
    }
  else
    {
      G95X_DUMP_SUBOBJ_NA (key, symbol_ref, &sr);
    }
}

static void
dump_local_assignment_ref_an (const void *sym, g95x_locus * xw, int opl)
{
  g95x_symbol_ref sr;

  g95x_get_code_substr (xw, sr.als, 2 * G95_MAX_SYMBOL_LEN);

  if (sr.als[0] == '\0')
    g95x_die ("");

  sr.sym = sym;
  sr.xw = xw;
  sr.cls = _S(G95X_S_ASSIGNMENT_REF);
  sr.strcmp = 0;
  sr.type = NULL;

  if (opl)
    {
      g95x_assignment1_ref uor;
      uor.sr = sr;
      g95x_opl (xw, &uor.xwo, "assignment(", ")");
      G95X_DUMP_SUBOBJ_AN (assignment1_ref, &uor);
    }
  else
    {
      G95X_DUMP_SUBOBJ_AN (symbol_ref, &sr);
    }
}

static void
dump_label_na (const char *key, g95_st_label ** label, g95x_locus * xw)
{
  g95x_label_ref lr;
  lr.addr = label;
  lr.label = *label;
  lr.xw = xw;
  G95X_DUMP_SUBOBJ_NA (key, label_ref, &lr);
}

static void
dump_label_an (g95_st_label ** label, g95x_locus * xw)
{
  g95x_label_ref lr;
  lr.addr = label;
  lr.label = *label;
  lr.xw = xw;
  G95X_DUMP_SUBOBJ_AN (label_ref, &lr);
}

static void
dump_expr_chst_na (const void *key, void *addr, g95x_locus * xw)
{
  g95x_expr_chst ch;
  ch.addr = addr;
  ch.where = xw;
  G95X_DUMP_SUBOBJ_NA (key, expr_chst, &ch);
}

#ifdef G95X_DUMP_BODY

static g95x_expr_chst *
get_temp_expr_chst (g95x_temp ** t, void *addr, g95x_locus * xw)
{
  g95x_temp *s = g95x_get_temp ();
  g95x_expr_chst *ec = g95_getmem (sizeof (g95x_expr_chst));
  s->next = *t;
  *t = s;
  s->obj = ec;

  ec->addr = addr;
  ec->where = xw;

  return ec;
}


static g95x_io_control_spec *
get_temp_io_control_spec_expr (g95x_temp ** t, g95_expr ** expr,
			       const char * iok, const char * name, const char * class)
{
  g95x_temp *s = g95x_get_temp ();
  g95x_io_control_spec *ics = g95_getmem (sizeof (g95x_io_control_spec));
  s->next = *t;
  *t = s;
  s->obj = ics;

  ics->type = G95X_CS_EXPR;
  ics->u.expr = expr;
  g95x_iok (&(*expr)->x.where, iok, &ics->where);
  ics->name = name;
  ics->class = class;

  return ics;
}


static g95x_io_control_spec *
get_temp_io_control_spec_expr_chst (g95x_temp ** t, g95_expr ** expr,
				    const char * iok, const char * name, g95x_locus * xw,
				    const char * class)
{
  g95x_temp *s = g95x_get_temp ();
  g95x_io_control_spec *ics = g95_getmem (sizeof (g95x_io_control_spec));
  s->next = *t;
  *t = s;
  s->obj = ics;

  ics->type = G95X_CS_EXCH;
  ics->u.expr = expr;
  g95x_iok (xw, iok, &ics->where);
  ics->name = name;
  ics->xw = xw;
  ics->class = class;

  return ics;
}


static g95x_io_control_spec *
get_temp_io_control_spec_label (g95x_temp ** t, g95_st_label ** label,
				const char * iok, const char * name, g95x_locus * xw,
				const char * class)
{
  g95x_temp *s = g95x_get_temp ();
  g95x_io_control_spec *ics = g95_getmem (sizeof (g95x_io_control_spec));
  s->next = *t;
  *t = s;
  s->obj = ics;

  ics->type = G95X_CS_LABL;
  ics->u.label = label;
  ics->xw = xw;
  g95x_iok (xw, iok, &ics->where);
  ics->name = name;
  ics->class = class;

  return ics;
}


static g95x_io_control_spec *
get_temp_io_control_spec_symbol (g95x_temp ** t, g95_symbol ** symbol,
				 const char * iok, const char * name, g95x_locus * xw,
				 const char * class)
{
  g95x_temp *s = g95x_get_temp ();
  g95x_io_control_spec *ics = g95_getmem (sizeof (g95x_io_control_spec));
  s->next = *t;
  *t = s;
  s->obj = ics;

  ics->type = G95X_CS_SMBL;
  ics->u.symbol = symbol;
  g95x_iok (xw, iok, &ics->where);
  ics->xw = xw;
  ics->name = name;
  ics->class = class;

  return ics;
}

#endif


static void
dump_actual_arglist (g95_actual_arglist * k)
{
  g95x_item_locus *lst;
  lst = NULL;
  for (; k; k = k->next)
    if (k->u.expr || k->u.label)
      G95X_PSH_LOC_AN (actual_arg, k, lst);
  G95X_ADD_LST (_S(G95X_S_ACTUAL G95X_S_HN G95X_S_ARG G95X_S_HN G95X_S_LIST));
  dump_lst_an (lst);
  G95X_END_LST ();
}


static void
free_temp (g95x_temp * temp)
{
  g95x_temp *t;
  while (temp)
    {
      g95_free ((void *) temp->obj);
      t = temp->next;
      g95_free (temp);
      temp = t;
    }
}

static void
dump_statement_code (g95x_statement * sta)
{
  g95_forall_iterator *fa;
  g95_open *open;
  g95_alloc *b;
  g95_close *close;
  g95_filepos *position;
  g95_inquire *inquire;
  g95_dt *io;
  g95_code *p, code;
  g95x_item_locus *lst = NULL;
  g95x_temp *temp;

  temp = NULL;


  if (!sta->code)
    return;


  if ((sta->type == ST_ASSIGNMENT) && (sta->code->type == EXEC_CALL))
    {
      /* This means that our assignment has been replaced
         by a call; hence it is an overloaded assignment.
         so we fake an EXEC_ASSIGN code */

      code = *(sta->code);
      p = &code;

      p->type = EXEC_ASSIGN;
      p->expr = p->ext.sub.actual->u.expr;
      p->expr2 = p->ext.sub.actual->next->u.expr;
    }
  else
    {
      p = sta->code;
    }


  if (sta->type == ST_ARITHMETIC_IF)
    {
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr);
      if (p->label)
	dump_label_na (_S(G95X_S_LABEL G95X_S_HN G95X_S_1), &p->label,
		       &p->x.label_where);
      if (p->label2)
	dump_label_na (_S(G95X_S_LABEL G95X_S_HN G95X_S_2), &p->label2,
		       &p->x.label2_where);
      if (p->label3)
	dump_label_na (_S(G95X_S_LABEL G95X_S_HN G95X_S_3), &p->label3,
		       &p->x.label3_where);
      return;
    }

  switch (sta->type)
    {
    case ST_SIMPLE_IF:
    case ST_WHERE:
      if (p->expr)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr);
      else if (p->block && p->block->expr)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->block->expr);
      else
	g95x_die ("");
      return;
      break;
    case ST_NULLIFY:
      p->expr2 = NULL;
      break;
    default:
      break;
    }


  /* computed goto */
  if ((sta->type == ST_GOTO) && (p->type == EXEC_SELECT))
    {
      g95_code *q;
      G95X_ADD_LST (_S(G95X_S_LABEL G95X_S_HN G95X_S_LIST));
      for (q = p->block; q; q = q->block)
	dump_label_an (&q->next->label, &q->next->x.label_where);
      G95X_END_LST ();
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr2);
      return;
    }

  switch (p->type)
    {
    case EXEC_NOP:
      break;

    case EXEC_CONTINUE:
      break;

    case EXEC_AC_ASSIGN:
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr);
      break;

    case EXEC_ASSIGN:
    case EXEC_POINTER_ASSIGN:

      if (sta->type == ST_NULLIFY)
	{
	  g95_code *c;
	  int n = 1;
	  G95X_ADD_LST (_S(G95X_S_POINTER G95X_S_HN G95X_S_OBJECT G95X_S_HN
			G95X_S_LIST));
	  for (c = p;; c = c->next, n++)
	    {
	      G95X_DUMP_SUBOBJ_AN (expr, c->expr);
	      if (n == sta->u.nullify.n)
		break;
	    }
	  G95X_END_LST ();

	}
      else
	{
	  const void *q;

	  if (p->expr)
            {
              if (p->type == EXEC_ASSIGN)
                G95X_DUMP_SUBOBJ_NA (_S(G95X_S_VARIABLE), expr, p->expr);
              else
                G95X_DUMP_SUBOBJ_NA (_S(G95X_S_POINTER G95X_S_HN G95X_S_OBJECT), expr, p->expr);
            }

	  if (g95x_option.xml_defn)
	    if (p->sym)		/* this is an overloaded assignment */
	      G95X_ADD_ATT_OBJ (_S(G95X_S_PROCEDURE), p->sym);

	  sta->code->x.call_where.lb1 = p->expr->x.where.lb2;
	  sta->code->x.call_where.column1 = p->expr->x.where.column2;
	  sta->code->x.call_where.lb2 = p->expr2->x.where.lb1;
	  sta->code->x.call_where.column2 = p->expr2->x.where.column1;

	  q = g95x_resolve_intrinsic_operator (INTRINSIC_ASSIGN);
	  dump_local_assignment_ref_na (_S(G95X_S_ASSIGNMENT), q,
					&sta->code->x.call_where, 0);

	  if (p->expr2)
            {
              if (p->type == EXEC_ASSIGN)
	        G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr2);
              else
	        G95X_DUMP_SUBOBJ_NA (_S(G95X_S_TARGET), expr, p->expr2);
            }

	}
      break;

    case EXEC_GOTO:
      if (p->label)
	dump_label_na (_S(G95X_S_LABEL), &p->label, &p->x.label_where);
      else
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LABEL), expr, p->expr);

      break;

    case EXEC_CALL:
      {

	if (sta->type == ST_CALL)
	  {
	    if (p->x.generic)
	      {			/* generic call */

		if (g95x_option.xml_defn)
		  G95X_ADD_ATT_OBJ (_S(G95X_S_SUBROUTINE), p->sym);

		dump_local_generic_ref_na (_S(G95X_S_GENERIC),
					   g95x_resolve_generic (p->ext.sub.
								 name),
					   &p->x.call_where);
	      }
	    else if (p->sym)
	      {			/* ordinary call */
		dump_global_symbol_ref_na (_S(G95X_S_SUBROUTINE), p->sym,
					   &p->x.call_where, 0);
	      }
	    else if (p->x.symbol)
	      {			/* intrinsic call */
		dump_local_generic_ref_na (_S(G95X_S_GENERIC),
					   g95x_resolve_generic (p->ext.sub.
								 name),
					   &p->x.call_where);

		if (g95x_option.xml_defn)
		  G95X_ADD_ATT_OBJ (_S(G95X_S_FUNCTION), p->ext.sub.isym);

	      }
	    else if (p->ext.sub.pointer)
	      {
		G95X_DUMP_SUBOBJ_NA (_S(G95X_S_SUBROUTINE), expr,
				     p->ext.sub.pointer);
	      }
	    else
	      {
		g95x_die ("");
	      }
	  }

	dump_actual_arglist (p->ext.sub.actual);

	break;
      }
    case EXEC_RETURN:
      if (p->expr)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr);
      break;

    case EXEC_STOP:
    case EXEC_ALL_STOP:
    case EXEC_PAUSE:

      if (p->expr)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_STOP G95X_S_HN G95X_S_CODE), expr,
			     p->expr);
      else if (p->ext.stop_code > 0)
        {
          g95x_stop_code sc;
          g95x_locus xwi;
          char code[32];
          sprintf (code, "%d", p->ext.stop_code);
          xwi.lb2 = xwi.lb1 = sta->where.lb2;
          xwi.column2 = xwi.column1 = sta->where.column2;
          g95x_opl (&xwi, &sc.where, code, "");
          sc.code = p->ext.stop_code;
          G95X_DUMP_SUBOBJ_NA (_S(G95X_S_STOP G95X_S_HN G95X_S_CODE),
                               stop_code, &sc);
        }

      break;

    case EXEC_IF:
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr);

      break;

    case EXEC_SELECT:

      if (p->expr)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CASE G95X_S_HN G95X_S_EXPR), expr,
			     p->expr);
      else if (p->expr2)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CASE G95X_S_HN G95X_S_EXPR), expr,
			     p->expr2);
      else
	{
	  g95_expr *lo = p->ext.case_list->low;
	  g95_expr *hi = p->ext.case_list->high;
	  G95X_FULLIFY (lo);
	  G95X_FULLIFY (hi);
	  if (lo && hi)
	    {
	      if (lo != hi)
		{
		  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CASE G95X_S_HN G95X_S_LOWER
				       G95X_S_HN G95X_S_BOUND), expr, lo);
		  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CASE G95X_S_HN G95X_S_UPPER
				       G95X_S_HN G95X_S_BOUND), expr, hi);
		}
	      else
		{
		  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CASE G95X_S_HN G95X_S_VALUE),
				       expr, lo);
		}
	    }
	}


      break;

    case EXEC_WHERE:
      if (p->block)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_MASK G95X_S_HN G95X_S_EXPR), expr,
			     p->block->expr);

      break;

    case EXEC_FORALL:
      G95X_ADD_LST (_S(G95X_S_FORALL G95X_S_HN G95X_S_TRIPLET G95X_S_HN
		    G95X_S_SPEC G95X_S_HN G95X_S_LIST));
      for (fa = p->ext.forall_iterator; fa; fa = fa->next)
	G95X_DUMP_SUBOBJ_AN (forall_iterator, fa);
      G95X_END_LST ();
      if (p->expr)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_MASK), expr, p->expr);

      break;

    case EXEC_DO:
      if (p->label)
	dump_label_na (_S(G95X_S_LABEL), &p->label, &p->x.label_where);
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ITERATOR), iterator, p->ext.iterator);

      break;

    case EXEC_DO_WHILE:
      /* Changed transform_while */
      if (p->label)
	dump_label_na (_S(G95X_S_LABEL), &p->label, &p->x.label_where);
      if (p->expr)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr);

      break;

    case EXEC_EXIT:
    case EXEC_CYCLE:
      break;

    case EXEC_DEALLOCATE:
    case EXEC_ALLOCATE:

      if (p->type == EXEC_ALLOCATE)
        G95X_ADD_LST (_S(G95X_S_ALLOCATION G95X_S_HN G95X_S_LIST));
      else
        G95X_ADD_LST (_S(G95X_S_DEALLOCATION G95X_S_HN G95X_S_LIST));

      for (b = p->ext.alloc_list; b; b = b->next)
	G95X_DUMP_SUBOBJ_AN (alloc, b);
      G95X_END_LST ();

      if (p->expr)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_STAT), expr, p->expr);

      break;

    case EXEC_OPEN:
      open = p->ext.open;


#define _piovA( iok, key, obj, att, cls ) \
  if( g95x_valid_expr( obj->att ) )  \
    _pioaA( iok, key, obj, att, cls )

#define _pioaA( iok, key, obj, att, cls ) \
  if( obj->att ) \
  G95X_PSH_LOC_AN( io_control_spec, \
    get_temp_io_control_spec_expr( &temp, \
    &obj->att, iok, _S(key), _S(cls) ), lst )

#define _piolA( iok, key, obj, att, cls ) \
  if( obj->att ) \
  G95X_PSH_LOC_AN( io_control_spec, \
    get_temp_io_control_spec_label( &temp, &obj->att, \
    iok, _S(key), &obj->x.att##_where, _S(cls) ), lst )

#define _piocA( iok, key, obj, att, cls ) \
  G95X_PSH_LOC_AN( io_control_spec, \
    get_temp_io_control_spec_expr_chst( &temp, &obj->att, \
    iok, _S(key), &obj->x.att##_where, _S(cls) ), lst )

#define _piosA( iok, key, obj, att, cls ) \
  G95X_PSH_LOC_AN( io_control_spec, \
    get_temp_io_control_spec_symbol( &temp, &obj->att, \
    iok, _S(key), &obj->x.att##_where, _S(cls) ), lst )

#define _piovB( obj, att, cls ) _piovA( #att, _s_##att, obj, att, cls )
#define _pioaB( obj, att, cls ) _pioaA( #att, _s_##att, obj, att, cls )
#define _piolB( obj, att, cls ) _piolA( #att, _s_##att, obj, att, cls )
#define _piocB( obj, att, cls ) _piocA( #att, _s_##att, obj, att, cls )
#define _piosB( obj, att, cls ) _piosA( #att, _s_##att, obj, att, cls )

#define _piov( obj, att ) _piovB( obj, att, _s_##obj G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC )
#define _pioa( obj, att ) _pioaB( obj, att, _s_##obj G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC )
#define _piol( obj, att ) _piolB( obj, att, _s_##obj G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC )
#define _pioc( obj, att ) _piocB( obj, att, _s_##obj G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC )
#define _pios( obj, att ) _piosB( obj, att, _s_##obj G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC )

#define _s_access        G95X_S_ACCESS
#define _s_action        G95X_S_ACTION
#define _s_advance       G95X_S_ADVANCE
#define _s_blank         G95X_S_BLANK
#define _s_close         G95X_S_CLOSE
#define _s_decimal       G95X_S_DECIMAL
#define _s_delim         G95X_S_DELIM
#define _s_direct        G95X_S_DIRECT
#define _s_end           G95X_S_END
#define _s_eor           G95X_S_EOR
#define _s_err           G95X_S_ERROR
#define _s_exist         G95X_S_EXIST
#define _s_file          G95X_S_FILE
#define _s_formatted     G95X_S_FORMATTED
#define _s_form          G95X_S_FORM
#define _s_id            G95X_S_ID
#define _s_io            G95X_S_IO
#define _s_iolength      G95X_S_IOLENGTH
#define _s_iomsg         G95X_S_IOMSG
#define _s_iostat        G95X_S_IOSTAT
#define _s_named         G95X_S_NAMED
#define _s_name          G95X_S_NAME_IO
#define _s_nextrec       G95X_S_NEXTREC
#define _s_number        G95X_S_NUMBER
#define _s_opened        G95X_S_OPENED
#define _s_open          G95X_S_OPEN
#define _s_pad           G95X_S_PAD
#define _s_pos           G95X_S_POS
#define _s_position      G95X_S_POSITION
#define _s_read          G95X_S_READ
#define _s_readwrite     G95X_S_READWRITE
#define _s_rec           G95X_S_REC
#define _s_recl          G95X_S_RECL
#define _s_sequential    G95X_S_SEQUENTIAL
#define _s_size          G95X_S_SIZE
#define _s_status        G95X_S_STATUS
#define _s_unformatted   G95X_S_UNFORMATTED
#define _s_unit          G95X_S_UNIT
#define _s_write         G95X_S_WRITE


      _pioa (open, unit);
      _pioa (open, iostat);
      _pioa (open, file);
      _pioa (open, status);
      _pioa (open, access);
      _pioa (open, form);
      _pioa (open, recl);
      _pioa (open, blank);
      _pioa (open, position);
      _pioa (open, action);
      _pioa (open, delim);
      _pioa (open, pad);
      _pioa (open, decimal);

      _piol (open, err);

      G95X_ADD_LST (_S(G95X_S_OPEN G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC
		    G95X_S_HN G95X_S_LIST));
      dump_lst_an (lst);
      G95X_END_LST ();
      break;

    case EXEC_CLOSE:
      close = p->ext.close;

      _pioa (close, unit);
      _pioa (close, iostat);
      _pioa (close, status);

      _piol (close, err);

      G95X_ADD_LST (_S(G95X_S_CLOSE G95X_S_HN G95X_S_CONTROL G95X_S_HN
		    G95X_S_SPEC G95X_S_HN G95X_S_LIST));
      dump_lst_an (lst);
      G95X_END_LST ();
      break;

    case EXEC_BACKSPACE:
    case EXEC_ENDFILE:
    case EXEC_REWIND:

      {
        char buffer[128];
        char * c = &buffer[0];
        g95x_get_code_substr (&sta->where, buffer, sizeof (buffer)-1);
        /* Skip label if any */
        for (; isdigit (*c); c++);
        /* Skip text */
        for (; isalpha (*c); c++);

        G95X_ADD_LST (_S(G95X_S_POSITION G95X_S_HN G95X_S_SPEC G95X_S_HN
        	    G95X_S_LIST));
        if (*c == '(')
          {
            position = p->ext.filepos;

            _pioaB (position, unit, G95X_S_POSITION G95X_S_HN G95X_S_SPEC);
            _pioaB (position, iostat, G95X_S_POSITION G95X_S_HN G95X_S_SPEC);
            _piolB (position, err, G95X_S_POSITION G95X_S_HN G95X_S_SPEC);

            dump_lst_an (lst);
          }
        else
          {
            g95_expr * unit;
            unit = p->ext.filepos->unit;
            G95X_DUMP_SUBOBJ_AN (position_spec_unit, unit);
          }
        G95X_END_LST ();
      }

      break;

    case EXEC_INQUIRE:
      inquire = p->ext.inquire;

      _pioaB (inquire, unit, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, file, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, iostat, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, exist, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, opened, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, number, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, named, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, name, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, access, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, sequential, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, direct, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, form, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, formatted, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, unformatted, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, recl, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, nextrec, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, blank, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, position, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, action, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, read, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, write, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, readwrite, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, delim, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, pad, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, iolength, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _pioaB (inquire, size, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);
      _piolB (inquire, err, G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);

      _pioaA ("position", G95X_S_POSITION, inquire, pos,
	      G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC);

      G95X_ADD_LST (_S(G95X_S_INQUIRE G95X_S_HN G95X_S_SPEC G95X_S_HN
		    G95X_S_LIST));
      dump_lst_an (lst);
      G95X_END_LST ();

      break;

    case EXEC_IOLENGTH:
      G95X_PSH_LOC_AN (io_control_spec,
		       get_temp_io_control_spec_expr (&temp, &p->expr,
                                                      "iolength", 
						      _S(G95X_S_IOLENGTH),
						      _S(G95X_S_INQUIRE G95X_S_HN
						      G95X_S_SPEC)), lst);
      goto print_transfers;

    case EXEC_READ:
    case EXEC_WRITE:
      {
	g95_code *c;
	int fmt = 0;
	io = p->ext.dt;

	if (g95x_valid_expr (io->io_unit))
	  {
	    _pioaA ("unit", G95X_S_UNIT, io, io_unit,
		    G95X_S_IO G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC);
	  }
	else if ((!sta->code->x.is_print) && io->x.io_unit_where.lb1)
	  {
	    _piocA ("unit", G95X_S_UNIT, io, io_unit,
		    G95X_S_IO G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC);
	  }

	if (g95x_valid_expr (io->format_expr) && (io->format_label != NULL))
	  if (io->format_label->value > 0)
	    g95x_die ("");

	if (g95x_valid_expr (io->format_expr))
	  {
	    _pioaA ("fmt", G95X_S_FMT, io, format_expr,
		    G95X_S_IO G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC);
	    fmt = 1;
	  }
	else
	  {
	    if (io->format_label != NULL)
	      if (io->format_label->value > 0)
		{
		  _piolA ("fmt", G95X_S_FMT, io, format_label,
			  G95X_S_IO G95X_S_HN G95X_S_CONTROL G95X_S_HN
			  G95X_S_SPEC);
		  fmt = 1;
		}
	  }
	if ((fmt == 0)
	    && (!io->namelist)
	    && (!g95x_valid_expr (io->rec))
	    && (io->x.format_expr_where.lb1 != NULL))
	  {
	    _piocA ("fmt", G95X_S_FMT, io, format_expr,
		    G95X_S_IO G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC);
	  }

	if (io->namelist)
	  _piosA ("nml", G95X_S_NAMELIST, io, namelist,
		  G95X_S_IO G95X_S_HN G95X_S_CONTROL G95X_S_HN G95X_S_SPEC);

	_piov (io, iostat);
	_piov (io, size);
	_piov (io, rec);
	_piov (io, advance);
	_piov (io, pos);
	_piov (io, decimal);

      print_transfers:{

	  for (c = p->next; c; c = c->next)
	    {
	      if (c->type == EXEC_DT_END)
		{
		  io = c->ext.dt;
		  if (io != NULL)
		    {
		      _piol (io, err);
		      _piol (io, end);
		      _piol (io, eor);
		    }
		  break;
		}
	    }

	  G95X_ADD_LST (_S(G95X_S_IO G95X_S_HN G95X_S_CONTROL G95X_S_HN
			G95X_S_SPEC G95X_S_HN G95X_S_LIST));
	  dump_lst_an (lst);
	  G95X_END_LST ();

	  if (p->type == EXEC_READ)
	    G95X_ADD_LST (_S(G95X_S_INPUT G95X_S_HN G95X_S_ITEM G95X_S_HN
			  G95X_S_LIST));
	  else if (p->type == EXEC_WRITE)
	    G95X_ADD_LST (_S(G95X_S_OUTPUT G95X_S_HN G95X_S_ITEM G95X_S_HN
			  G95X_S_LIST));
	  else if (p->type == EXEC_IOLENGTH)
	    G95X_ADD_LST (_S(G95X_S_OUTPUT G95X_S_HN G95X_S_ITEM G95X_S_HN
			  G95X_S_LIST));
	  else
	    g95x_die ("");
	  for (c = p->next; c; c = c->next)
	    {
	      switch (c->type)
		{
		case EXEC_TRANSFER:
		  G95X_DUMP_SUBOBJ_AN (expr, c->expr);
		  break;
		case EXEC_DO:
		  G95X_DUMP_SUBOBJ_AN (io_var, c);
		  break;
		default:
		  break;
		}
	      if (c->type == EXEC_DT_END)
		{
		  G95X_END_LST ();
		  break;
		}
	    }
	}
      }

      break;

    case EXEC_FLUSH:
      {
	g95_flush *flush = p->ext.flush;
	_piovB (flush, unit, G95X_S_FLUSH G95X_S_HN G95X_S_SPEC);
	_piovB (flush, iostat, G95X_S_FLUSH G95X_S_HN G95X_S_SPEC);
	_piovB (flush, iomsg, G95X_S_FLUSH G95X_S_HN G95X_S_SPEC);
	_piolB (flush, err, G95X_S_FLUSH G95X_S_HN G95X_S_SPEC);
	G95X_ADD_LST (_S(G95X_S_FLUSH G95X_S_HN G95X_S_SPEC G95X_S_HN
		      G95X_S_LIST));
	dump_lst_an (lst);
	G95X_END_LST ();
      }
      break;

    case EXEC_WAIT:
      {
	g95_wait *wait = p->ext.wait;
	_piovB (wait, unit, G95X_S_WAIT G95X_S_HN G95X_S_SPEC);
	_piovB (wait, id, G95X_S_WAIT G95X_S_HN G95X_S_SPEC);
	_piovB (wait, iostat, G95X_S_WAIT G95X_S_HN G95X_S_SPEC);
	_piovB (wait, iomsg, G95X_S_WAIT G95X_S_HN G95X_S_SPEC);
	_piolB (wait, err, G95X_S_WAIT G95X_S_HN G95X_S_SPEC);
	_piolB (wait, end, G95X_S_WAIT G95X_S_HN G95X_S_SPEC);
	_piolB (wait, eor, G95X_S_WAIT G95X_S_HN G95X_S_SPEC);
	G95X_ADD_LST (_S(G95X_S_WAIT G95X_S_HN G95X_S_SPEC G95X_S_HN
		      G95X_S_LIST));
	dump_lst_an (lst);
	G95X_END_LST ();
      }
      break;

    case EXEC_TRANSFER:
    case EXEC_DT_END:
      g95x_die ("Attempt to print TRANSFER/DT_END code\n");
      break;

    case EXEC_ENTRY:
      break;

    case EXEC_LABEL_ASSIGN:
      dump_label_na (_S(G95X_S_LABEL), &p->label, &p->x.label_where);
      dump_global_symbol_ref_na (_S(G95X_S_SYMBOL), p->sym, &p->x.assign_where,
				 0);
      break;

    case EXEC_SYNC_ALL:
    case EXEC_SYNC_TEAM:
    case EXEC_SYNC_IMAGES:
    case EXEC_SYNC_MEMORY:

      if (g95x_valid_expr (p->expr))
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, p->expr);
      if (g95x_valid_expr (p->ext.sync.stat))
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_STAT), expr, p->ext.sync.stat);
      if (g95x_valid_expr (p->ext.sync.errmsg))
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ERROR G95X_S_HN G95X_S_MESSAGE), expr,
			     p->ext.sync.errmsg);

      break;

    case EXEC_CRITICAL:
      break;

    case EXEC_AC_START:
    case EXEC_WHERE_ASSIGN:

    default:
      g95x_die ("Unknown code\n");

    }


  free_temp (temp);

#undef _piovA
#undef _pioaA
#undef _piolA
#undef _piocA
#undef _piosA

#undef _piovB
#undef _pioaB
#undef _piolB
#undef _piocB
#undef _piosB

#undef _piov
#undef _pioa
#undef _piol
#undef _pioc
#undef _pios


}

static void
dump_statement_IMPORT (g95x_statement * sta)
{
  g95x_voidp_list *sl0 = g95x_set_voidp_prev (sta->u.import.decl_list), *sl;
  G95X_ADD_LST (_S(G95X_S_IMPORT G95X_S_HN G95X_S_NAME G95X_S_HN G95X_S_LIST));
  for (sl = sl0; sl; sl = sl->prev)
    dump_global_symbol_ref_an (sl->u.symbol, &sl->where);
  G95X_END_LST ();
}

static void
dump_statement_MODULE_PROC (g95x_statement * sta)
{
  g95x_voidp_list *sl0 = g95x_set_voidp_prev (sta->u.modproc.decl_list), *sl;
  G95X_ADD_LST (_S(G95X_S_PROCEDURE G95X_S_HN G95X_S_NAME G95X_S_HN G95X_S_LIST));
  for (sl = sl0; sl; sl = sl->prev)
    dump_global_symbol_ref_an (sl->u.symbol, &sl->where);
  G95X_END_LST ();
}

static void
dump_statement_SIMPLE_IF (g95x_statement * sta)
{
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ACTION G95X_S_HN G95X_S_STATEMENT), statement,
		       sta->next);
}

static void
dump_statement_WHERE (g95x_statement * sta)
{
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ASSIGNMENT G95X_S_HN G95X_S_STATEMENT),
		       statement, sta->next);
}

static void
dump_statement_FORALL (g95x_statement * sta)
{
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ASSIGNMENT G95X_S_HN G95X_S_STATEMENT),
		       statement, sta->next);
}

static void
dump_statement_USE (g95x_statement * sta)
{
  g95_use_rename *list;
  dump_global_symbol_ref_na (_S(G95X_S_MODULE G95X_S_HN G95X_S_NAME),
			     sta->u.use.module.u.symbol,
			     &sta->u.use.module.where, 0);
  G95X_ADD_LST (_S(G95X_S_RENAME G95X_S_HN G95X_S_LIST));
  for (list = sta->u.use.rename_list; list; list = list->next)
    G95X_DUMP_SUBOBJ_AN (rename, list);
  G95X_END_LST ();
}

static void
dump_statement_DIMENSION (g95x_statement * sta)
{
  g95x_voidp_list *sl0 =
    g95x_set_voidp_prev (sta->u.data_decl.decl_list), *sl;

  G95X_ADD_LST (_S(G95X_S_ENTITY G95X_S_HN G95X_S_DECL G95X_S_HN G95X_S_LIST));
  for (sl = sl0; sl; sl = sl->prev)
    G95X_DUMP_SUBOBJ_AN (entity_decl, sl);
  G95X_END_LST ();

}

static void
dump_statement_END_INTERFACE (g95x_statement * sta)
{
  const void *p;
  g95x_statement *sta1 =
    sta->u.interface.st_interface ? sta->u.interface.st_interface : sta;
  g95x_locus *xw;
  int i;

  switch (sta1->u.interface.info.type)
    {
    case INTERFACE_ABSTRACT:
    case INTERFACE_NAMELESS:
      break;
    case INTERFACE_GENERIC:
      for (i = sta->n_ext_locs - 1; i >= 0; i--)
	if (sta->ext_locs[i].type & ~G95X_LOCUS_LANG)
	  {
	    xw = &sta->ext_locs[i].loc;
	    dump_local_generic_ref_na (_S(G95X_S_GENERIC),
				       sta1->u.interface.info.generic, xw);
	    break;
	  }
      break;
    case INTERFACE_INTRINSIC_OP:
      p = g95x_resolve_intrinsic_operator (sta1->u.interface.info.op);
      for (i = sta->n_ext_locs - 1; i >= 0; i--)
	if (sta->ext_locs[i].type & ~G95X_LOCUS_LANG)
	  {
	    xw = &sta->ext_locs[i].loc;
	    if (sta1->u.interface.info.op == INTRINSIC_ASSIGN)
	      dump_local_assignment_ref_na (_S(G95X_S_ASSIGNMENT), p, xw, 1);
	    else
	      dump_local_intrinsic_operator_ref_na (_S(G95X_S_OPERATOR), p, xw,
						    1);
	    break;
	  }
      break;
    case INTERFACE_USER_OP:
      for (i = sta->n_ext_locs - 1; i >= 0; i--)
	if (sta->ext_locs[i].type & ~G95X_LOCUS_LANG)
	  {
	    xw = &sta->ext_locs[i].loc;
	    dump_local_user_operator_ref_na (_S(G95X_S_OPERATOR),
					     sta1->u.interface.info.uop, xw,
					     1);
	    break;
	  }
      break;
    default:
      g95x_die ("Unexpected interface");
    }

}

static void
dump_statement_INTERFACE (g95x_statement * sta)
{
  G95X_ADD_ATT_STR (_S(G95X_S_INTERFACE G95X_S_HN G95X_S_TYPE),
		    g95x_interface_type (sta->u.interface.info.type));
  dump_statement_END_INTERFACE (sta);
}

static void
dump_statement_ATTR_DECL (g95x_statement * sta)
{
  g95_namespace *ns = sta->ns;
  symbol_attribute *attr = &sta->u.data_decl.attr;
  g95_expr *bind = sta->u.data_decl.bind;

  if (attr->allocatable || attr->pointer || attr->dimension || attr->target)
    return dump_statement_DIMENSION (sta);



  g95x_voidp_list *sl0 =
    g95x_set_voidp_prev (sta->u.data_decl.decl_list), *sl;

  if (attr->bind && bind)
    {
      G95X_FULLIFY (bind);
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_BIND G95X_S_HN G95X_S_NAME), expr, bind);
    }

  if (attr->intent != INTENT_UNKNOWN)
    G95X_ADD_LST (
      _S(G95X_S_DUMMY G95X_S_HN G95X_S_ARG G95X_S_HN G95X_S_NAME G95X_S_HN
      G95X_S_LIST));
  else if (attr->access != ACCESS_UNKNOWN)
    G95X_ADD_LST (
      _S(G95X_S_ACCESS G95X_S_HN G95X_S_ID G95X_S_HN G95X_S_LIST));
  else if (attr->external)
    G95X_ADD_LST (
      _S(G95X_S_EXTERNAL G95X_S_HN G95X_S_NAME G95X_S_HN G95X_S_LIST));
  else if (attr->intrinsic)
    G95X_ADD_LST (
      _S(G95X_S_INTRINSIC G95X_S_HN G95X_S_PROCEDURE G95X_S_HN G95X_S_NAME
      G95X_S_HN G95X_S_LIST));
  else if (attr->volatile_)
    G95X_ADD_LST (_S(G95X_S_OBJECT G95X_S_HN G95X_S_NAME G95X_S_HN G95X_S_LIST));
  else if (attr->optional)
    G95X_ADD_LST (
      _S(G95X_S_DUMMY G95X_S_HN G95X_S_ARG G95X_S_HN G95X_S_NAME G95X_S_HN
      G95X_S_LIST));
  else if (attr->save)
    G95X_ADD_LST (
      _S(G95X_S_SAVED G95X_S_HN G95X_S_ENTITY G95X_S_HN G95X_S_LIST));
  else if (attr->value)
    G95X_ADD_LST (
      _S(G95X_S_DUMMY G95X_S_HN G95X_S_ARG G95X_S_HN G95X_S_NAME G95X_S_HN
      G95X_S_LIST));
  else if (attr->async)
    G95X_ADD_LST (_S(G95X_S_OBJECT G95X_S_HN G95X_S_NAME G95X_S_HN G95X_S_LIST));
  else if (attr->bind)
    G95X_ADD_LST (_S(G95X_S_BIND G95X_S_HN G95X_S_ENTITY G95X_S_HN G95X_S_LIST));
  else
    g95x_die ("");

  for (sl = sl0; sl; sl = sl->prev)
    {
      g95_symtree *st;

      switch (sl->type)
	{
	case G95X_VOIDP_INTROP_IDX:
	  if (sl->u.introp_idx == INTRINSIC_ASSIGN)
	    dump_local_assignment_ref_an (g95x_resolve_intrinsic_operator
					  (sl->u.introp_idx), &sl->where, 1);
	  else
	    dump_local_intrinsic_operator_ref_an
	      (g95x_resolve_intrinsic_operator (sl->u.introp_idx), &sl->where,
	       1);
	  break;
	case G95X_VOIDP_SYMBOL:
	  if (attr->access != ACCESS_UNKNOWN)
	    {
	      g95_symbol *sym = sl->u.symbol;
	      if ((st = g95_find_generic (sym->name, ns)))
		dump_local_generic_ref_an (st, &sl->where);
	      else
		dump_local_symbol_ref_an (sym, &sl->where);
	    }
	  else if (sl->u.symbol->attr.intrinsic)
	    {
	      g95_symbol *sym = sl->u.symbol;
	      if (sym->ts.type != BT_PROCEDURE)
		dump_local_generic_ref_an (g95_find_function (sym->name),
					   &sl->where);
	      else
		dump_local_generic_ref_an (g95_find_subroutine (sym->name),
					   &sl->where);
	    }
	  else
	    {
	      dump_local_symbol_ref_an (sl->u.symbol, &sl->where);
	    }
	  break;
	case G95X_VOIDP_COMMON:
	  dump_local_common_ref_an (sl->u.common, &sl->where);
	  break;
	case G95X_VOIDP_INTROP:
	  dump_local_intrinsic_operator_ref_an (sl->u.introp, &sl->where, 1);
	  break;
	case G95X_VOIDP_USEROP:
	  dump_local_user_operator_ref_an (sl->u.userop->n.uop, &sl->where,
					   1);
	  break;
	case G95X_VOIDP_COMPNT:
	  dump_local_component_ref_an (sl->u.component, &sl->where,
                                       &sl->u.component->x.derived->ts);
	  break;
	case G95X_VOIDP_GENERIC:
	  dump_local_generic_ref_an (sl->u.generic, &sl->where);
	  break;
	case G95X_VOIDP_UNKNOWN:
	  g95x_die ("");
	  break;
	}

    }
  G95X_END_LST ();

}


static void
dump_statement_ENUM (g95x_statement * sta)
{
  if (sta->enum_bindc)
    G95X_ADD_ATT_STR (_S(G95X_S_BIND), _S(G95X_S_C));
}

static void
get_tklc (g95_typespec * ts, const void **typeid, g95_expr ** kind,
	  g95_expr ** length, g95_charlen ** cl)
{
  g95_charlen *clc = ts->cl;
  *length = NULL;
  *kind = NULL;
  *cl = NULL;
  *typeid = g95x_type_id (ts);
  if (clc)
    {
      if (clc->x.alias)
	clc = clc->x.alias;
      *length = clc->length;
      if (ts->type == BT_CHARACTER)
	*cl = clc;
      if (g95x_valid_expr (*length))
	G95X_FULLIFY (*length);
      else
	*length = NULL;
    }
  else
    {
      *length = NULL;
    }
  if (ts->x.kind)
    *kind = ts->x.kind->kind;
  else
    *kind = NULL;
}

static void
dump_statement_IMPLICIT (g95x_statement * sta)
{
  g95x_implicit_spec *is;
  G95X_ADD_LST (_S(G95X_S_IMPLICIT G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
  for (is = sta->u.implicit.list_is; is; is = is->next)
    G95X_DUMP_SUBOBJ_AN (implicit_spec, is);
  G95X_END_LST ();
}

static void
dump_statement_PARAMETER (g95x_statement * sta)
{
  g95x_voidp_list *sl, *sl0 =
    g95x_set_voidp_prev (sta->u.data_decl.decl_list);
  G95X_ADD_LST (_S(G95X_S_NAMED G95X_S_HN G95X_S_CONSTANT G95X_S_HN G95X_S_DEF
		G95X_S_HN G95X_S_LIST));
  for (sl = sl0; sl; sl = sl->prev)
    G95X_DUMP_SUBOBJ_AN (named_constant_def, sl);
  G95X_END_LST ();
}

static void
dump_statement_COMMON (g95x_statement * sta)
{
  g95x_voidp_list *sl0 = g95x_set_voidp_prev (sta->u.common.common_list), *sl;

  G95X_ADD_LST (_S(G95X_S_COMMON G95X_S_HN G95X_S_BLOCK G95X_S_HN G95X_S_DECL
		G95X_S_HN G95X_S_LIST));
  sl = sl0;
  for (; sl; sl = sl->prev)
    if (sl->type == G95X_VOIDP_COMMON)
      G95X_DUMP_SUBOBJ_AN (common_block_decl, sl);
  G95X_END_LST ();

}

#ifdef G95X_DUMP_BODY

static g95x_attr_spec *
get_temp_attr_spec_result (g95x_temp ** temp, g95_symbol * sym,
			   g95x_locus * xwr, g95x_statement * sta)
{
  g95x_temp *t = g95x_get_temp ();
  g95x_attr_spec *attr;
  g95x_locus xw;

  t->next = *temp;
  *temp = t;

  t->obj = attr = g95x_get_attr_spec ();

  attr->name = _S(G95X_S_RESULT);
  attr->result = sym;
  attr->result_where = xwr;


  g95x_locate_attr (sta, "result", &attr->where);

  xw.lb1 = xwr->lb1;
  xw.column1 = xwr->column1;
  xw.lb2 = sta->where.lb2;
  xw.column2 = sta->where.column2;

  g95x_match_word (&xw, ")");

  attr->where.lb2 = xw.lb2;
  attr->where.column2 = xw.column2;


  return attr;
}

static g95x_attr_spec *
get_temp_attr_spec (g95x_temp ** temp, const char *name,
		    const char *key, g95x_statement * sta)
{
  g95x_temp *t = g95x_get_temp ();
  g95x_attr_spec *attr;

  t->next = *temp;
  *temp = t;

  t->obj = attr = g95x_get_attr_spec ();

  attr->name = name;
  g95x_locate_attr (sta, key, &attr->where);

  return attr;
}

static g95x_attr_spec *
get_temp_attr_spec_as (g95x_temp ** temp, g95x_statement * sta)
{
  g95x_temp *t = g95x_get_temp ();
  g95x_attr_spec *attr;
  g95_array_spec *as = (g95_array_spec *) sta->u.data_decl.as;
  g95_coarray_spec *cas = (g95_coarray_spec *) sta->u.data_decl.cas;

  t->next = *temp;
  *temp = t;

  t->obj = attr = g95x_get_attr_spec ();

  attr->name = _S(G95X_S_DIMENSION);
  g95x_locate_attr (sta, "dimension", &attr->where);

  g95x_get_ac ((g95_array_spec *) sta->u.data_decl.as,
	       (g95_coarray_spec *) sta->u.data_decl.cas, &as, &cas);

  if (as == NULL)
    g95x_die ("");
  if (cas)
    g95x_die ("");

  attr->as = as;

  if (as)
    {
      attr->where.lb2 = as->x.where.lb2;
      attr->where.column2 = as->x.where.column2;
    }

  return attr;
}

static g95x_attr_spec *
get_temp_attr_spec_bind (g95x_temp ** temp, g95x_statement * sta,
			 g95_expr * bind)
{
  g95x_temp *t = g95x_get_temp ();
  g95x_attr_spec *attr;
  g95x_locus xw;

  t->next = *temp;
  *temp = t;

  t->obj = attr = g95x_get_attr_spec ();

  attr->name = _S(G95X_S_BIND);
  attr->bind = bind;

  g95x_locate_attr (sta, "bind", &attr->where);

  xw.lb1 = bind->x.where.lb1;
  xw.column1 = bind->x.where.column1;
  xw.lb2 = sta->where.lb2;
  xw.column2 = sta->where.column2;

  g95x_match_word (&xw, ")");

  attr->where.lb2 = xw.lb2;
  attr->where.column2 = xw.column2;

  return attr;
}

#endif

static void
print_statement_attributes (g95x_statement * sta, symbol_attribute * a,
			    g95_expr * bind, int what)
{
  g95x_temp *temp;
  g95x_item_locus *lst;


  temp = NULL;
  lst = NULL;

#define pattr_( str, key )                                   \
    G95X_PSH_LOC_AN(                                         \
      attr_spec,                                             \
      get_temp_attr_spec( &temp, _S(str), key, sta ),    \
      lst                                                    \
    )


#define pattrA( att, str, key ) \
  if( a->att )                  \
    pattr_( str, key )

#define pattrD( att, str ) pattrA( att, str, #att )

  if (what & G95X_DUMP_ATTR_DEC)
    {

      pattrD (allocatable, G95X_S_ALLOCATABLE);
      pattrD (external,    G95X_S_EXTERNAL   );
      pattrD (intrinsic,   G95X_S_INTRINSIC  );
      pattrD (optional,    G95X_S_OPTIONAL   );
      pattrD (pointer,     G95X_S_POINTER    );
      pattrD (save,        G95X_S_SAVE       );
      pattrD (target,      G95X_S_TARGET     );
      pattrD (value,       G95X_S_VALUE      );
      pattrD (elemental,   G95X_S_ELEMENTAL  );
      pattrD (pure,        G95X_S_PURE       );
      pattrD (recursive,   G95X_S_RECURSIVE  );

      pattrA (volatile_, G95X_S_VOLATILE,     "volatile"    );
      pattrA (async,     G95X_S_ASYNCHRONOUS, "asynchronous");

      if (a->dimension)
	G95X_PSH_LOC_AN (attr_spec, get_temp_attr_spec_as (&temp, sta), lst);

    }

  if (what & G95X_DUMP_ATTR_SUF)
    {
      if (a->bind)
	{
	  if (bind)
	    {
	      G95X_PSH_LOC_AN (attr_spec,
			       get_temp_attr_spec_bind (&temp, sta, bind),
			       lst);
	    }
	  else
	    {
	      pattrA (bind, G95X_S_BIND, G95X_S_BIND_C_);
	    }
	}
    }

  switch (a->access)
    {
    case ACCESS_PUBLIC:
      pattr_ (G95X_S_PUBLIC, "public");
      break;
    case ACCESS_PRIVATE:
      pattr_ (G95X_S_PRIVATE, "private");
      break;
    case ACCESS_UNKNOWN:
      break;
    }

  switch (a->intent)
    {
    case INTENT_IN:
      pattr_ (G95X_S_INTENT_IN_, "intent(in)");
      break;
    case INTENT_OUT:
      pattr_ (G95X_S_INTENT_OUT_, "intent(out)");
      break;
    case INTENT_INOUT:
      pattr_ (G95X_S_INTENT_INOUT_, "intent(inout)");
      break;
    case INTENT_UNKNOWN:
      break;
    }

  switch (a->proc)
    {
    case PROC_MODULE:
    case PROC_INTERNAL:
    case PROC_DUMMY:
    case PROC_ST_FUNCTION:
      break;
    case PROC_INTRINSIC:
      pattr_ (G95X_S_INTRINSIC, "intrinsic");
      break;
    case PROC_EXTERNAL:
      pattr_ (G95X_S_EXTERNAL, "external");
      break;
    case PROC_UNKNOWN:
      break;
    }
  switch (a->flavor)
    {
    case FL_PARAMETER:
      pattr_ (G95X_S_PARAMETER, "parameter");
      break;
    case FL_UNKNOWN:
      break;
    case FL_PROGRAM:
    case FL_BLOCK_DATA:
    case FL_MODULE:
    case FL_VARIABLE:
    case FL_LABEL:
    case FL_PROCEDURE:
    case FL_DERIVED:
    case FL_NAMELIST:
      break;
    }

  G95X_ADD_LST (_S(G95X_S_ATTR G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
  dump_lst_an (lst);
  G95X_END_LST ();

  free_temp (temp);



}

static void
print_attributes_list (symbol_attribute * a)
{
  G95X_ADD_LST (_S(G95X_S_ATTR G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
  switch (a->access)
    {
    case ACCESS_PUBLIC:
      G95X_PSH_LST_STR (_S(G95X_S_PUBLIC));
      break;
    case ACCESS_PRIVATE:
      G95X_PSH_LST_STR (_S(G95X_S_PRIVATE));
      break;
    case ACCESS_UNKNOWN:
      break;
    }

  switch (a->intent)
    {
    case INTENT_IN:
      G95X_PSH_LST_STR (_S(G95X_S_INTENT_IN_));
      break;
    case INTENT_OUT:
      G95X_PSH_LST_STR (_S(G95X_S_INTENT_OUT_));
      break;
    case INTENT_INOUT:
      G95X_PSH_LST_STR (_S(G95X_S_INTENT_INOUT_));
      break;
    case INTENT_UNKNOWN:
      break;
    }

  switch (a->proc)
    {
    case PROC_MODULE:
    case PROC_INTERNAL:
    case PROC_DUMMY:
    case PROC_ST_FUNCTION:
      break;
    case PROC_INTRINSIC:
      G95X_PSH_LST_STR (_S(G95X_S_INTRINSIC));
      break;
    case PROC_EXTERNAL:
      G95X_PSH_LST_STR (_S(G95X_S_EXTERNAL));
      break;
    case PROC_UNKNOWN:
      break;
    }
  switch (a->flavor)
    {
    case FL_PARAMETER:
      G95X_PSH_LST_STR (_S(G95X_S_PARAMETER));
      break;
    case FL_UNKNOWN:
      break;
    case FL_PROGRAM:
    case FL_BLOCK_DATA:
    case FL_MODULE:
    case FL_VARIABLE:
    case FL_LABEL:
    case FL_PROCEDURE:
    case FL_DERIVED:
    case FL_NAMELIST:
      break;
    }

#define print_attr( attr, attn ) if( a->attr ) \
  { G95X_PSH_LST_STR( #attn ); }

  print_attr (allocatable, ALLOCATABLE);
  print_attr (external, EXTERNAL);
  print_attr (intrinsic, INTRINSIC);
  print_attr (optional, OPTIONAL);
  print_attr (pointer, POINTER);
  print_attr (save, SAVE);
  print_attr (target, TARGET);
  print_attr (value, VALUE);
  print_attr (volatile_, VOLATILE);
  print_attr (async, ASYNCHRONOUS);
  print_attr (elemental, ELEMENTAL);
  print_attr (pure, PURE);
  print_attr (recursive, RECURSIVE);
  print_attr (bind, BIND);

#undef print_attr

  G95X_END_LST ();
}

static void
dump_statement_function_like (const char *block_name,
			      g95x_statement * sta, symbol_attribute * attr,
			      g95_expr * bind, g95_typespec * ts,
			      g95_symbol * result)
{
  g95_symbol *sym = sta->block;
  g95_formal_arglist *fa;
  g95x_temp *temp;
  g95x_item_locus *lst;

  temp = NULL;
  lst = NULL;


  if (attr)
    {
      if (attr->recursive)
	G95X_PSH_LOC_AN (prefix_spec,
			 get_temp_attr_spec (&temp, _S(G95X_S_RECURSIVE),
					     "recursive", sta), lst);
      if (attr->pure)
	G95X_PSH_LOC_AN (prefix_spec,
			 get_temp_attr_spec (&temp, _S(G95X_S_PURE), "pure",
					     sta), lst);
      if (attr->elemental)
	G95X_PSH_LOC_AN (prefix_spec,
			 get_temp_attr_spec (&temp, _S(G95X_S_ELEMENTAL),
					     "elemental", sta), lst);
    }

  if (ts)
    G95X_PSH_LOC_NA (_S(G95X_S_TYPE G95X_S_HN G95X_S_SPEC), typespec, ts, lst);

  G95X_ADD_LST (_S(G95X_S_PREFIX G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
  dump_lst_an (lst);
  G95X_END_LST ();
  free_temp (temp);

  dump_local_symbol_ref_na (block_name, sta->block, &sta->block->x.where);

  G95X_ADD_LST (_S(G95X_S_DUMMY G95X_S_HN G95X_S_ARG G95X_S_HN G95X_S_NAME
		G95X_S_HN G95X_S_LIST));
  for (fa = sym->formal; fa; fa = fa->next)
    G95X_DUMP_SUBOBJ_AN (formal_arg, fa);
  G95X_END_LST ();

  temp = NULL;
  lst = NULL;

  if (result && (result != sym))
    G95X_PSH_LOC_NA (_S(G95X_S_RESULT), attr_spec,
		     get_temp_attr_spec_result (&temp, result,
						&result->x.where, sta), lst);
  if (attr && attr->bind)
    {
      if (bind)
	G95X_PSH_LOC_NA (_S(G95X_S_BIND), attr_spec,
			 get_temp_attr_spec_bind (&temp, sta, bind), lst);
      else
	G95X_PSH_LOC_NA (_S(G95X_S_BIND), attr_spec,
			 get_temp_attr_spec (&temp, _S(G95X_S_BIND),
					     _S(G95X_S_BIND_C_), sta), lst);
    }

  dump_lst_na (lst);
  free_temp (temp);

}

static void
dump_statement_SUBROUTINE (g95x_statement * sta)
{
  g95_expr *bind = sta->block->x.bind;
  symbol_attribute *attr = &sta->u.subroutine.attr;

  G95X_FULLIFY (bind);

  dump_statement_function_like (_S(G95X_S_SUBROUTINE G95X_S_HN G95X_S_NAME),
				sta, attr, bind, NULL, NULL);
}

static void
dump_statement_END_SUBROUTINE (g95x_statement * sta)
{
  if (g95x_locus_valid (&sta->block->x.end_where))
    dump_local_symbol_ref_na (_S(G95X_S_SUBROUTINE G95X_S_HN G95X_S_NAME),
			      sta->block, &sta->block->x.end_where);
}

static void
dump_statement_ENTRY (g95x_statement * sta)
{
  g95_symbol *sym = sta->block;
  g95_expr *bind = sym->x.bind;
  g95_symbol *result = sym->result;

  G95X_FULLIFY (bind);

  dump_statement_function_like (_S(G95X_S_ENTRY G95X_S_HN G95X_S_NAME),
				sta, NULL, bind, NULL, result);
}

static void
dump_statement_FUNCTION (g95x_statement * sta)
{
  g95_symbol *sym = sta->block;
  symbol_attribute *attr = &sta->u.function.attr;
  g95_expr *bind = sym->x.bind;
  g95_symbol *result = sym->result;
  g95_typespec *ts = &sta->u.function.ts;

  G95X_FULLIFY (bind);

  dump_statement_function_like (_S(G95X_S_FUNCTION G95X_S_HN G95X_S_NAME),
				sta, attr, bind,
				ts->type != BT_UNKNOWN ? ts : NULL, result);
}

static void
dump_statement_END_FUNCTION (g95x_statement * sta)
{
  if (g95x_locus_valid (&sta->block->x.end_where))
    dump_local_symbol_ref_na (_S(G95X_S_FUNCTION G95X_S_HN G95X_S_NAME),
			      sta->block, &sta->block->x.end_where);
}

static void
dump_statement_STATEMENT_FUNCTION (g95x_statement * sta)
{
  dump_statement_function_like (_S(G95X_S_STATEMENT G95X_S_HN G95X_S_FUNCTION),
				sta, NULL, NULL, NULL, NULL);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_VALUE), expr, sta->block->value);
}


static void
dump_statement_MODULE (g95x_statement * sta)
{
  dump_local_symbol_ref_na (_S(G95X_S_MODULE G95X_S_HN G95X_S_NAME), sta->block,
			    &sta->block->x.where);
}

static void
dump_statement_END_MODULE (g95x_statement * sta)
{
  if (g95x_locus_valid (&sta->block->x.end_where))
    dump_local_symbol_ref_na (_S(G95X_S_MODULE G95X_S_HN G95X_S_NAME), sta->block,
			      &sta->block->x.end_where);
}

static void
dump_statement_DERIVED_DECL (g95x_statement * sta)
{
  print_statement_attributes (sta, &sta->u.data_decl.attr, NULL,
			      G95X_DUMP_ATTR_ALL);
  dump_local_symbol_ref_na (_S(G95X_S_TYPE G95X_S_HN G95X_S_NAME), sta->block,
			    &sta->block->x.where);
}

static void
dump_statement_END_TYPE (g95x_statement * sta)
{
  if (g95x_locus_valid (&sta->block->x.end_where))
    dump_local_symbol_ref_na (_S(G95X_S_TYPE G95X_S_HN G95X_S_NAME), sta->block,
			      &sta->block->x.end_where);
}

static void
dump_statement_BLOCK_DATA (g95x_statement * sta)
{
  g95_symbol *sym = sta->block;
  if (!strcmp (sym->name, BLANK_BLOCK_DATA_NAME))
    return;
  dump_local_symbol_ref_na (_S(G95X_S_BLOCK G95X_S_HN G95X_S_DATA G95X_S_HN
			    G95X_S_NAME), sta->block, &sta->block->x.where);
}

static void
dump_statement_END_BLOCK_DATA (g95x_statement * sta)
{
  g95_symbol *sym = sta->block;
  if (!strcmp (sym->name, BLANK_BLOCK_DATA_NAME))
    return;
  if (g95x_locus_valid (&sta->block->x.end_where))
    dump_local_symbol_ref_na (_S(G95X_S_BLOCK G95X_S_HN G95X_S_DATA G95X_S_HN
			      G95X_S_NAME), sta->block,
			      &sta->block->x.end_where);
}

static void
dump_statement_PROGRAM (g95x_statement * sta)
{
  dump_local_symbol_ref_na (_S(G95X_S_PROGRAM G95X_S_HN G95X_S_NAME), sta->block,
			    &sta->block->x.where);
}

static void
dump_statement_END_PROGRAM (g95x_statement * sta)
{
  g95_symbol *sym = sta->block;
  if (sym == NULL)
    return;
  if (g95x_locus_valid (&sta->block->x.end_where))
    dump_local_symbol_ref_na (_S(G95X_S_PROGRAM G95X_S_HN G95X_S_NAME),
			      sta->block, &sta->block->x.end_where);
}

static void
dump_statement_add_block (g95x_statement * sta, g95_symbol * block)
{
  int i;
  g95x_locus *xw;
  if (block == NULL)
    return;
  for (i = 0; i < sta->n_ext_locs; i++)
    if (sta->ext_locs[i].type & G95X_LOCUS_NAME)
      goto found;
  g95x_die ("");
found:
  xw = &sta->ext_locs[i].loc;
  dump_local_symbol_ref_na (_S(G95X_S_BLOCK G95X_S_HN G95X_S_NAME), block, xw);
}

static void
dump_statement_end_block (g95x_statement * sta, g95_symbol * block, int must)
{
  int i;
  g95x_locus *xw;
  if (block == NULL)
    return;
  for (i = sta->n_ext_locs - 1; i >= 0; i--)
    if (sta->ext_locs[i].type & G95X_LOCUS_NAME)
      goto found;
  if (must)
    g95x_die ("");
  else
    return;
found:
  xw = &sta->ext_locs[i].loc;
  dump_local_symbol_ref_na (_S(G95X_S_BLOCK G95X_S_HN G95X_S_NAME), block, xw);
}

static void
dump_statement_FORALL_BLOCK (g95x_statement * sta)
{
  dump_statement_add_block (sta, sta->block);
}

static void
dump_statement_END_FORALL (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->block, 1);
}

static void
dump_statement_WHERE_BLOCK (g95x_statement * sta)
{
  dump_statement_add_block (sta, sta->block);
}

static void
dump_statement_ELSEWHERE (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->block, 0);
}

static void
dump_statement_END_WHERE (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->block, 1);
}

static void
dump_statement_IF_BLOCK (g95x_statement * sta)
{
  dump_statement_add_block (sta, sta->block);
}

static void
dump_statement_ELSE (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->block, 0);
}

static void
dump_statement_ENDIF (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->block, 1);
}

static void
dump_statement_SELECT_CASE (g95x_statement * sta)
{
  dump_statement_add_block (sta, sta->block);
}

static void
dump_statement_CASE (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->block, 0);
}

static void
dump_statement_END_SELECT (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->block, 1);
}

static void
dump_statement_DO (g95x_statement * sta)
{
  dump_statement_add_block (sta, sta->block);
}

static void
dump_statement_CYCLE (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->code->sym, 1);
}

static void
dump_statement_EXIT (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->code->sym, 1);
}

static void
dump_statement_ENDDO (g95x_statement * sta)
{
  dump_statement_end_block (sta, sta->block, 1);
}

static void
dump_statement_EQUIVALENCE (g95x_statement * sta)
{
  g95_equiv *e;
  /* forward */
  for (e = sta->u.equiv.e1;; e = e->next)
    {
      if (e->next)
	e->next->x.prev = e;
      if (e->next == sta->u.equiv.e2)
	break;
    }
  /* now backward */
  G95X_ADD_LST (_S(G95X_S_EQUIVALENCE G95X_S_HN G95X_S_SET G95X_S_HN
		G95X_S_LIST));
  for (;; e = e->x.prev)
    {
      G95X_DUMP_SUBOBJ_AN (equivalence_set, e);
      if (e == sta->u.equiv.e1)
	break;
    }
  G95X_END_LST ();
}

static void
dump_statement_FORMAT (g95x_statement * sta)
{
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_FORMAT), expr, sta->u.format.here->format);
}

static void
dump_statement_ENUMERATOR (g95x_statement * sta)
{
  g95x_voidp_list *sl =
    g95x_set_voidp_prev (sta->u.enumerator.enumerator_list);

  G95X_ADD_LST (_S(G95X_S_ENUMERATION G95X_S_HN G95X_S_LIST));
  for (; sl; sl = sl->prev)
    G95X_DUMP_SUBOBJ_AN (enumeration, sl);
  G95X_END_LST ();
}


static void
dump_statement_DATA (g95x_statement * sta)
{
  g95_data *d;

  /* forward */
  for (d = sta->u.data.d1;; d = d->next)
    {
      if (d->next)
	d->next->x.prev = d;
      if (d->next == sta->u.data.d2)
	break;
    }
  /* now backward */
  G95X_ADD_LST (_S(G95X_S_DATA G95X_S_HN G95X_S_STATEMENT G95X_S_HN G95X_S_SET
		G95X_S_HN G95X_S_LIST));
  for (;; d = d->x.prev)
    {
      G95X_DUMP_SUBOBJ_AN (data_statement_set, d);
      if (d == sta->u.data.d1)
	break;
    }
  G95X_END_LST ();
}

static void
dump_statement_DATA_DECL (g95x_statement * sta)
{
  g95x_voidp_list *f, *f0;
  g95_expr *bind = sta->u.data_decl.bind;
  G95X_FULLIFY (bind);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_TYPE G95X_S_HN G95X_S_SPEC), typespec,
		       &sta->u.data_decl.ts);

  print_statement_attributes (sta, &sta->u.data_decl.attr, bind,
			      G95X_DUMP_ATTR_ALL);

  f0 = g95x_set_voidp_prev (sta->u.data_decl.decl_list);
  if (sta->in_derived)
    {
      G95X_ADD_LST (_S(G95X_S_COMPONENT G95X_S_HN G95X_S_DECL G95X_S_HN
		    G95X_S_LIST));
      for (f = f0; f; f = f->prev)
	G95X_DUMP_SUBOBJ_AN (component_decl, f);
      G95X_END_LST ();
    }
  else
    {
      G95X_ADD_LST (_S(G95X_S_ENTITY G95X_S_HN G95X_S_DECL G95X_S_HN
		    G95X_S_LIST));
      for (f = f0; f; f = f->prev)
	G95X_DUMP_SUBOBJ_AN (entity_decl, f);
      G95X_END_LST ();
    }






}

static void
dump_statement_NAMELIST (g95x_statement * sta)
{
  g95x_voidp_list *sl0 =
    g95x_set_voidp_prev (sta->u.namelist.namelist_list), *sl;
  G95X_ADD_LST (_S(G95X_S_NAMELIST G95X_S_HN G95X_S_GROUP G95X_S_HN G95X_S_LIST));
  for (sl = sl0; sl; sl = sl->prev)
    if (sl->u.symbol->attr.flavor == FL_NAMELIST)
      G95X_DUMP_SUBOBJ_AN (namelist_group, sl);
  G95X_END_LST ();
}

#endif

#ifndef INTF

static void
dump_attr_spec (g95x_attr_spec * attr)
{
  G95X_DUMP_WHERE (&attr->where);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), attr->name);
  if (attr->as)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC), array_spec,
			 attr->as);
  if (attr->bind)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_BIND G95X_S_HN G95X_S_NAME), expr, attr->bind);
  if (attr->result)
    dump_local_symbol_ref_na (_S(G95X_S_RESULT G95X_S_HN G95X_S_NAME),
			      attr->result, attr->result_where);
}

#endif

void
G95X_DUMP_FCT (stop_code, g95x_stop_code * c)
#ifdef INTF
;
#else
{
  g95_typespec ts;
  memset (&ts, '\0', sizeof (ts));
  ts.type = BT_INTEGER;
  G95X_ADD_OBJ_T_NI (_S(G95X_S_EXPR), _S(G95X_S_LITERAL), c);
  G95X_DUMP_ATTR_FCT (stop_code, c);
  G95X_ADD_ATT_INT (_S(G95X_S_VALUE), c->code);

  G95X_ADD_ATT_STR (_S(G95X_S_CONSTANT G95X_S_HN G95X_S_TYPE),
  		    g95x_type_name (&ts));

  G95X_END_OBJ (stop_code, c);
}
#endif

void
G95X_DUMP_FCT (alloc, g95_alloc * a)
#ifdef INTF
;
#else
{
  g95_ref *ref = NULL, **pref = NULL;

  if (a->x.dealloc)
    G95X_ADD_OBJ (_S(G95X_S_DEALLOCATION), a);
  else
    G95X_ADD_OBJ (_S(G95X_S_ALLOCATION), a);

  G95X_DUMP_ATTR_FCT (alloc, a);
  
  for (pref = &a->expr->ref; (*pref) && (*pref)->next; pref = &(*pref)->next);
  if (pref && (*pref) && ((*pref)->type == REF_ARRAY))
    {
      ref = *pref; (*pref) = NULL;
    } 
  else
    {
      pref = NULL;
    }

  if (a->x.dealloc)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_DEALLOCATE G95X_S_HN G95X_S_OBJECT), expr, a->expr);
  else
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ALLOCATE G95X_S_HN G95X_S_OBJECT), expr, a->expr);

  if (pref)
    *pref = ref;

  if (a->rank && !a->x.dealloc) 
  {
    g95x_alloc_spec ss;
    int i;
    g95x_locus* xw = &ss.where;
    G95X_ADD_LST (_S(G95X_S_ALLOCATE G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
    for (i = 0; i < a->rank; i++) {
      ss.lb = a->lower[i];
      ss.ub = a->upper[i];
      *xw = ss.ub->x.where;
      if (ss.lb) 
      {
        xw->lb1 = ss.lb->x.where.lb1; xw->column1 = ss.lb->x.where.column1;
      }
      G95X_DUMP_SUBOBJ_AN (alloc_spec, &ss);
    }
    G95X_END_LST ();
  }
  if (a->corank)
  {
    g95x_co_alloc_spec ss;
    int i;
    g95x_locus* xw = &ss.where;
    G95X_ADD_LST (_S(G95X_S_CO G95X_S_HN G95X_S_ALLOCATE G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
    for (i = 0; i < a->corank-1; i++) {
      ss.lb = g95x_valid_expr(a->lower[i+a->rank]) ? a->lower[i+a->rank] : NULL;
      ss.ub = a->upper[i+a->rank];
      *xw = ss.ub->x.where;
      if (ss.lb)
      {
        xw->lb1 = ss.lb->x.where.lb1; xw->column1 = ss.lb->x.where.column1;
      }
      G95X_DUMP_SUBOBJ_AN (co_alloc_spec, &ss);
    }
    g95x_locus xwS = a->upper[a->corank+a->rank-2]->x.where;
    g95x_expr_chst ch;
    xwS.column1 = xwS.column2; xwS.lb1 = xwS.lb2;
    g95x_opl (&xwS, &xwS, "", ",");
    xwS.column1 = xwS.column2; xwS.lb1 = xwS.lb2;
    g95x_opl (&xwS, &xwS, "", "*");

    ch.addr = &xwS;
    ch.where = &xwS;

    G95X_DUMP_SUBOBJ_AN (expr_chst, &ch);
    
    G95X_END_LST ();
  }

  G95X_END_OBJ (alloc, a);
}
#endif

void
G95X_DUMP_FCT (letter_spec, g95x_letter_spec * ls)
#ifdef INTF
;
#else
{
  char tmp[16];

  G95X_ADD_OBJ (_S(G95X_S_LETTER G95X_S_HN G95X_S_SPEC), ls);
  G95X_DUMP_ATTR_FCT (letter_spec, ls);
  G95X_DUMP_WHERE (&ls->where);

  if (ls->c2 > 0)
    sprintf (tmp, "%c-%c", ls->c1, ls->c2);
  else
    sprintf (tmp, "%c", ls->c1);

  G95X_ADD_ATT_STR (_S(G95X_S_VALUE), tmp);

  G95X_END_OBJ (letter_spec, ls);
}
#endif

void
G95X_DUMP_FCT (implicit_spec, g95x_implicit_spec * is)
#ifdef INTF
;
#else
{
  g95x_locus xw;
  int i, n;
  g95x_letter_spec ls[26];

  i = 0;
  n = 0;
  ls[0].c1 = '\0';

  G95X_ADD_OBJ_NI (_S(G95X_S_IMPLICIT G95X_S_HN G95X_S_SPEC), is);
  G95X_DUMP_ATTR_FCT (implicit_spec, is);
  G95X_DUMP_WHERE (&is->where);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_TYPE G95X_S_HN G95X_S_SPEC), typespec, &is->ts);
  xw.lb1 = is->ts.x.where.lb2;
  xw.column1 = is->ts.x.where.column2;
  xw.lb2 = is->where.lb2;
  xw.column2 = is->where.column2;

#ifdef G95X_DUMP_BODY
  g95x_parse_letter_spec_list (&xw, &n, &ls[0]);
  G95X_ADD_LST (_S(G95X_S_LETTER G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
  for (i = 0; i < n; i++)
    G95X_DUMP_SUBOBJ_AN (letter_spec, &ls[i]);
  G95X_END_LST ();
#endif

  G95X_END_OBJ (implicit_spec, is);
}
#endif

void
G95X_DUMP_FCT (prefix_spec, g95x_attr_spec * attr)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_PREFIX G95X_S_HN G95X_S_SPEC), attr);
  G95X_DUMP_ATTR_FCT (prefix_spec, attr);
  dump_attr_spec (attr);
  G95X_END_OBJ (prefix_spec, attr);
}
#endif

void
G95X_DUMP_FCT (attr_spec, g95x_attr_spec * attr)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_ATTR G95X_S_HN G95X_S_SPEC), attr);
  G95X_DUMP_ATTR_FCT (attr_spec, attr);
  dump_attr_spec (attr);
  G95X_END_OBJ (attr_spec, attr);
}
#endif

void
G95X_DUMP_FCT (symbol_ref, g95x_symbol_ref * c)
#ifdef INTF
;
#else
{
  int cd;

  G95X_ADD_OBJ (c->cls, c->xw);
  G95X_DUMP_ATTR_FCT (symbol_ref, c);


  if (g95x_option.xml_defn)
    {
      if (c->strcmp)
	G95X_ADD_ATT_ALS (_S(G95X_S_SYMBOL), c->als, c->sym);
      else
	G95X_ADD_ATT_OBJ (_S(G95X_S_SYMBOL), c->sym);
    }
  else
    {
      G95X_ADD_ATT_STR (_S(G95X_S_NAME), c->als);
    }

  if (c->type)
    G95X_ADD_ATT_STR (_S(G95X_S_TYPE), c->type);



  G95X_DUMP_WHERE (c->xw);

  cd = g95x_put_txt1_cd;
  g95x_put_txt1_cd = 1;
  G95X_END_OBJ (symbol_ref, c);
  g95x_put_txt1_cd = cd;
}
#endif


void
G95X_DUMP_FCT (io_control_spec, g95x_io_control_spec * c)
#ifdef INTF
;
#else
{
  g95x_label_ref lr;
  G95X_ADD_OBJ_NI (c->class, c->u.p);
  G95X_DUMP_ATTR_FCT (io_control_spec, c);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), c->name);
  G95X_DUMP_WHERE (&c->where);

  switch (c->type)
    {
    case G95X_CS_EXPR:
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_VALUE), expr, *(c->u.expr));
      break;
    case G95X_CS_LABL:
      lr.addr = c->xw;
      lr.label = *(c->u.label);
      lr.xw = c->xw;
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LABEL), label_ref, &lr);
      break;
    case G95X_CS_SMBL:
      dump_global_symbol_ref_na (_S(G95X_S_SYMBOL), *(c->u.symbol), c->xw, 0);
      break;
    case G95X_CS_EXCH:
      dump_expr_chst_na (_S(G95X_S_VALUE), c->xw, c->xw);
      break;
    }

  G95X_END_OBJ (io_control_spec, c);
}
#endif

void
G95X_DUMP_FCT (label_ref, g95x_label_ref * c)
#ifdef INTF
;
#else
{
  int cd;

  G95X_ADD_OBJ (_S(G95X_S_LABEL G95X_S_HN G95X_S_REFERENCE), c->addr);
  G95X_DUMP_ATTR_FCT (label_ref, c);
  if (g95x_option.xml_defn)
    G95X_ADD_ATT_OBJ (_S(G95X_S_LABEL), c->label);
  else
    G95X_ADD_ATT_INT (_S(G95X_S_LABEL), c->label->value);

  G95X_DUMP_WHERE (c->xw);

  cd = g95x_put_txt1_cd;
  g95x_put_txt1_cd = 1;
  G95X_END_OBJ (label_ref, c);
  g95x_put_txt1_cd = cd;
}
#endif



void
G95X_DUMP_FCT (equivalence_set, g95_equiv * f)
#ifdef INTF
;
#else
{
  g95_expr *g;
  g95_equiv *e;
  G95X_ADD_OBJ_NI (_S(G95X_S_EQUIVALENCE G95X_S_HN G95X_S_SET), f);
  G95X_ADD_LST (_S(G95X_S_EQUIVALENCE G95X_S_HN G95X_S_OBJECT G95X_S_HN
		G95X_S_LIST));
  for (e = f; e; e = e->eq)
    {
      g = e->expr;
      G95X_FULLIFY (g);
      G95X_DUMP_SUBOBJ_AN (expr, g);
    }
  G95X_END_LST ();
  G95X_END_OBJ (equivalence_set, f);
}
#endif



void
G95X_DUMP_FCT (keyword, g95_actual_arglist * k)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_NI (_S(G95X_S_KEYWORD), &k->start);
  G95X_DUMP_ATTR_FCT (keyword, k);
  G95X_DUMP_WHERE (&k->x.name_where);
  G95X_END_OBJ (keyword, k);
}
#endif

void
G95X_DUMP_FCT (actual_arg, g95_actual_arglist * k)
#ifdef INTF
;
#else
{

  if ((k->type != ARG_ALT_RETURN) && (k->u.expr == NULL))
    return;

  G95X_ADD_OBJ_NI (_S(G95X_S_ACTUAL G95X_S_HN G95X_S_ARG), k);

  G95X_DUMP_ATTR_FCT (actual_arg, k);
  G95X_DUMP_WHERE (&k->x.where);

  if (k->name)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_KEYWORD), keyword, k);
  if (k->type == ARG_ALT_RETURN)
    dump_label_na (_S(G95X_S_ALT G95X_S_HN G95X_S_RETURN G95X_S_HN G95X_S_LABEL),
		   &k->u.label, &k->x.label_where);
  else if (k->u.expr)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_VALUE), expr, k->u.expr);
  G95X_END_OBJ (actual_arg, k);
}
#endif

void
G95X_DUMP_FCT (section_subscript, g95x_section_subscript * ss)
#ifdef INTF
;
#else
{
  g95_ref *d = ss->ref;
  g95_array_ref *a = &d->u.ar;
  int t = ss->index;

  G95X_ADD_OBJ_NI (_S(G95X_S_SECTION G95X_S_HN G95X_S_SUBSCRIPT), &a->start[t]);
  G95X_DUMP_ATTR_FCT (section_subscript, ss);
  G95X_DUMP_WHERE (&a->x.section_where[t]);

  switch (a->dimen_type[t])
    {
    case DIMEN_RANGE:
      if (a->start[t])
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LOWER G95X_S_HN G95X_S_BOUND), expr,
			     a->start[t]);
      if (a->end[t])
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_UPPER G95X_S_HN G95X_S_BOUND), expr,
			     a->end[t]);
      if (a->stride[t])
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_STRIDE), expr, a->stride[t]);
      break;
    case DIMEN_ELEMENT:
      if (a->start[t])
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_SUBSCRIPT), expr, a->start[t]);
      break;
    case DIMEN_VECTOR:
      if (a->start[t])
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_VECTOR), expr, a->start[t]);
      break;
    default:
      g95x_die ("");
      break;
    }
  G95X_END_OBJ (section_subscript, ss);
}
#endif




#ifndef INTF

static int
eq_intf_info (g95x_intf_info * iif1, g95x_intf_info * iif2)
{
  if (iif1->type != iif2->type)
    return 0;
  if (strcmp (iif1->module, iif2->module))
    return 0;
  switch (iif1->type)
    {
    case INTERFACE_GENERIC:
      if (strcmp (iif1->u.generic_name, iif2->u.generic_name))
	return 0;
      break;
    case INTERFACE_INTRINSIC_OP:
      if (iif1->u.iop_idx != iif2->u.iop_idx)
	return 0;
      break;
    case INTERFACE_USER_OP:
      if (strcmp (iif1->u.uop_name, iif2->u.uop_name))
	return 0;
      break;
    default:
      g95x_die ("");
    }
  return 1;
}


static g95_interface *
rewind_intf (g95_interface * intf)
{
  g95_interface *intf1 = intf->x.head;
  if (intf1 == NULL)
    return intf;
  for (; intf1; intf1 = intf1->next)
    {
      if (eq_intf_info (intf1->x.intf_info, intf->x.intf_info))
	return intf1;
    }
  g95x_die ("");
  return NULL;
}

#endif


#ifndef INTF

static void
dump_use_local (g95_use_rename * d)
{

  int type = d->x.type;
  if (type < 0)
    {
      dump_global_symbol_ref_na (_S(G95X_S_LOCAL G95X_S_HN G95X_S_NAME),
				 d->x.u_local.sym, &d->x.local_where, 0);
    }
  else
    {
      switch (d->x.type)
	{
	case INTERFACE_GENERIC:
	  dump_local_generic_ref_na (_S(G95X_S_LOCAL G95X_S_HN G95X_S_NAME),
				     d->x.u_local.st_gnrc, &d->x.local_where);
	  break;
	case INTERFACE_INTRINSIC_OP:
	  dump_local_intrinsic_operator_ref_na (_S(G95X_S_LOCAL G95X_S_HN
						G95X_S_NAME),
						d->x.u_local.iop_intf,
						&d->x.local_where, 1);
	  break;
	case INTERFACE_USER_OP:
	  dump_local_user_operator_ref_na (_S(G95X_S_LOCAL G95X_S_HN G95X_S_NAME),
					   d->x.u_local.uop,
					   &d->x.local_where, 1);
	  break;
	case INTERFACE_NAMELESS:
	case INTERFACE_ABSTRACT:
	  g95x_die ("");
	}
    }

}


static void
dump_use_name (g95_use_rename * d, int global)
{
  g95_interface *intf;

  int type = d->x.type;
  if (type < 0)
    {
      dump_global_symbol_ref_na (_S(G95X_S_USE G95X_S_HN G95X_S_NAME),
				 d->x.u_use.sym, &d->x.use_where, global);
    }
  else
    {
      switch (d->x.type)
	{
	case INTERFACE_GENERIC:
	  intf = rewind_intf (d->x.u_use.gnrc);
	  dump_local_generic_ref_na (_S(G95X_S_GENERIC), intf->x.intf_info,
				     &d->x.use_where);
	  break;
	case INTERFACE_INTRINSIC_OP:
	  intf = rewind_intf (d->x.u_use.iop_intf);
	  if (intf->x.intf_info->u.iop_idx == INTRINSIC_ASSIGN)
	    dump_local_assignment_ref_na (_S(G95X_S_ASSIGNMENT),
					  intf->x.intf_info, &d->x.use_where,
					  1);
	  else
	    dump_local_intrinsic_operator_ref_na (_S(G95X_S_OPERATOR),
						  intf->x.intf_info,
						  &d->x.use_where, 1);
	  break;
	case INTERFACE_USER_OP:
	  intf = rewind_intf (d->x.u_use.uop);
	  dump_local_user_operator_ref_na (_S(G95X_S_OPERATOR), intf->x.intf_info,
					   &d->x.use_where, 1);
	  break;
	case INTERFACE_NAMELESS:
	case INTERFACE_ABSTRACT:
	  g95x_die ("");
	}
    }
}

#endif


void
G95X_DUMP_FCT (rename, g95_use_rename * d)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_NI (_S(G95X_S_RENAME), d);
  G95X_DUMP_ATTR_FCT (rename, d);
  G95X_DUMP_WHERE (&d->x.where);
  if (d->local_name[0])
    {
      dump_use_local (d);
      dump_use_name (d, 1);
    }
  else
    {
      dump_use_name (d, 0);
    }
  G95X_END_OBJ (rename, d);
}
#endif


void
G95X_DUMP_FCT (ref, g95_ref * d)
#ifdef INTF
;
#else
{
  g95_array_ref *a;
  g95_coarray_ref *car;
  int t;
  g95x_section_subscript ss;
  int i;

  G95X_ADD_OBJ_T_NI (_S(G95X_S_REFERENCE), g95x_ref_type (d), d);

  G95X_DUMP_ATTR_FCT (ref, d);

  G95X_DUMP_WHERE (&d->x.where);

  switch (d->type)
    {
    case REF_ARRAY:
      {
	a = &d->u.ar;
	switch (a->type)
	  {

	  case AR_ELEMENT:
	  case AR_SECTION:

	    ss.ref = d;

	    for (i = 0; i < a->dimen; i++)
	      if (a->start[i] || a->end[i] || a->stride[i])
		{
		  ss.ref = d;
		  G95X_ADD_LST (_S(G95X_S_SECTION G95X_S_HN G95X_S_SUBSCRIPT
				G95X_S_HN G95X_S_LIST));
		  for (ss.index = 0, t = 0; t < a->dimen; ss.index++, t++)
		    G95X_DUMP_SUBOBJ_AN (section_subscript, &ss);
		  G95X_END_LST ();
		  break;
		}

	    break;

	  case AR_FULL:
	    break;

	  case AR_UNKNOWN:
	    g95x_die ("Unknown array ref");
	    break;

	  }
	break;
      }
    case REF_COMPONENT:
      {
	g95x_locus *xr;
	int nxr;
	if (g95x_option.xml_defn)
	  {
	    G95X_ADD_ATT_OBJ (_S(G95X_S_COMPONENT G95X_S_HN G95X_S_TYPE),
			      d->u.c.sym);
	  }
	d->x.where_obj = d->x.where;
	g95x_refine_location (&d->x.where_obj, &xr, &nxr);
	d->x.where_obj.column1 = xr[0].column1 + 1;
	d->x.where_obj.lb1 = xr[0].lb1;
	dump_local_component_ref_na (_S(G95X_S_COMPONENT), d->u.c.component,
				     &d->x.where_obj, d->x.ts);
      }
      break;
    case REF_SUBSTRING:

      if (d->u.ss.start)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LOWER), expr, d->u.ss.start);

      if (d->u.ss.end)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_UPPER), expr, d->u.ss.end);

      break;

    case REF_COARRAY:
      car = &d->u.car;
      G95X_ADD_LST (_S(G95X_S_SECTION G95X_S_HN G95X_S_CO G95X_S_HN
		    G95X_S_SUBSCRIPT G95X_S_HN G95X_S_LIST));
      for (t = 0; t < car->dimen; t++)
	G95X_DUMP_SUBOBJ_AN (section_cosubscript, &car->element[t]);
      G95X_END_LST ();
      break;

    default:

      g95x_die ("Unknown reference type");
    }

  G95X_END_OBJ (ref, d);

  return;
}
#endif

void
G95X_DUMP_FCT (section_cosubscript, g95_expr ** e)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_NI (_S(G95X_S_SECTION G95X_S_HN G95X_S_CO G95X_S_HN
		   G95X_S_SUBSCRIPT), e);
  G95X_DUMP_ATTR_FCT (section_cosubscript, e);
  G95X_DUMP_WHERE (&(*e)->x.where);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_SUBSCRIPT), expr, *e);
  G95X_END_OBJ (section_cosubscript, e);
}
#endif

void
G95X_DUMP_FCT (iterator, g95_iterator * it)
#ifdef INTF
;
#else
{
  g95_expr *var, *start, *end, *step;

  G95X_ADD_OBJ_NI (_S(G95X_S_ITERATOR), it);

  G95X_DUMP_ATTR_FCT (iterator, it);

  var = it->var;
  start = it->start;
  end = it->end;
  step = it->step;

  G95X_FULLIFY (var);
  G95X_FULLIFY (start);
  G95X_FULLIFY (end);
  G95X_FULLIFY (step);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_VARIABLE), expr, var);
  if (start)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_START), expr, start);
  if (end)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END), expr, end);
  if (g95x_valid_expr (step))
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_STEP), expr, step);

  G95X_END_OBJ (iterator, it);

}
#endif

void
G95X_DUMP_FCT (constructor, g95_constructor * c)
#ifdef INTF
;
#else
{

  G95X_ADD_OBJ_NI (_S(G95X_S_CONSTRUCTOR), c);

  G95X_DUMP_ATTR_FCT (constructor, c);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, c->expr);

  if (c->iterator)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ITERATOR), iterator, c->iterator);

  G95X_END_OBJ (constructor, c);

}
#endif

#ifndef INTF

static void
dump_constant_location (g95_expr * h)
{
  g95x_locus xw;
  g95x_locus xws;
  int seen = 0;
  g95_linebuf *lb;
  int column;

#ifndef G95X_DUMP_LOCUS
  return;
#endif

  if (h->ts.x.kind == NULL)
    return;

  xw = h->x.where;

  column = xw.column1;
  lb = xw.lb1;


  while (1)
    {
      if (lb->file != xw.lb1->file)
	break;
      if ((lb == xw.lb2) && ((column >= xw.column2) || (column > lb->size)))
	break;
      if (column > lb->size)
	{
	  column = 1;
	  lb = lb->next;
	}

      if (h->ts.type == BT_CHARACTER)
	{
	  if (lb->mask[column - 1] & G95X_CHAR_STRG)
	    {
	      if (seen++)
		{
		  xws.lb2 = lb;
		  xws.column2 = column;
		}
	      else
		{
		  xws.lb1 = lb;
		  xws.column1 = column;
		}
	    }
	}
      else
	{
	  if (seen++)
	    {
	      xws.lb2 = lb;
	      xws.column2 = column;
	      if ((lb->line[column - 1] == '_')
		  && (lb->mask[column - 1] & G95X_CHAR_CODE))
		break;
	    }
	  else
	    {
	      xws.lb1 = lb;
	      xws.column1 = column;
	    }
	}

      column++;
    }

  if (h->ts.type == BT_CHARACTER)
    xws.column2++;


  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CONSTANT G95X_S_HN G95X_S_LOCATION),
		       constant_location, &xws);

}

#endif


void
G95X_DUMP_FCT (constant_location, g95x_locus * xw)
#ifdef INTF
;
#else
{
  g95x_locus *xr, *xr0;
  int nxr;
  g95x_locus_section ls;
  int i;


  g95x_refine_location (xw, &xr, &nxr);

  xr0 = &xr[0];
  G95X_ADD_OBJ (_S(G95X_S_LOCUS), &xr0->lb1->line[xr0->column1 - 1]);

  G95X_ADD_LST (_S(G95X_S_SECTION G95X_S_HN G95X_S_LIST));
  for (i = 0; i < nxr; i++)
    {
      ls.xr = &xr[i];
      G95X_DUMP_SUBOBJ_AN (locus_section, &ls);
    }
  G95X_END_LST ();

  G95X_END_OBJ (constant_location, xw);

}
#endif

#ifndef INTF

static void
dump_expr_op (g95_expr * h, g95_expr * op1, g95_expr * op2)
{
  g95x_locus *xw = &h->x.where_op;
  *xw = h->x.where;
  if (op2)
    {
      xw->lb1 = op1->x.where.lb2;
      xw->column1 = op1->x.where.column2;
      xw->lb2 = op2->x.where.lb1;
      xw->column2 = op2->x.where.column1;
    }
  else
    {
      xw->lb2 = op1->x.where.lb1;
      xw->column2 = op1->x.where.column1;
    }

  if (op2 != NULL)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_OPERAND1), expr, op1);

  if (h->type == EXPR_OP)
    {
      dump_local_intrinsic_operator_ref_na (_S(G95X_S_OPERATOR),
					    g95x_resolve_intrinsic_operator
					    (h->value.op.operator ), xw, 0);
    }
  else if (h->x.op == INTRINSIC_USER)
    {
      dump_local_user_operator_ref_na (_S(G95X_S_OPERATOR), h->x.uop->n.uop, xw,
				       0);
    }
  else
    {
      dump_local_intrinsic_operator_ref_na (_S(G95X_S_OPERATOR),
					    g95x_resolve_intrinsic_operator
					    (h->x.op), xw, 0);
    }

  if (g95x_option.xml_defn)
    if (h->type == EXPR_FUNCTION)
      G95X_ADD_ATT_OBJ (_S(G95X_S_PROCEDURE), h->symbol);

  if (op2)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_OPERAND2), expr, op2);
  else
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_OPERAND1), expr, op1);

}

#endif


void
G95X_DUMP_FCT (component_spec, g95_constructor * c)
#ifdef INTF
;
#else
{
  g95_expr *e;

  G95X_ADD_OBJ_NI (_S(G95X_S_COMPONENT G95X_S_HN G95X_S_SPEC), c);

  G95X_DUMP_ATTR_FCT (component_spec, c);

  if (g95x_locus_valid(&c->x.where_kw))
    dump_local_component_ref_na (_S(G95X_S_COMPONENT), c->x.comp, &c->x.where_kw, NULL);

  e = c->expr;
  G95X_FULLIFY (e);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, e);

  G95X_END_OBJ (component_spec, c);
}
#endif

void
G95X_DUMP_FCT (expr, g95_expr * h)
#ifdef INTF
;
#else
{
  g95_ref *r;
  g95_constructor *c;
  g95_actual_arglist *k;


  G95X_ADD_OBJ_T_NI (_S(G95X_S_EXPR), g95x_expr_type (h), h);

  G95X_DUMP_ATTR_FCT (expr, h);

  G95X_DUMP_WHERE (&h->x.where);


  switch (h->type)
    {
    case EXPR_STRUCTURE:

      h->x.where_sy = h->x.where;
      g95x_locate_name (&h->x.where_sy);
      dump_global_symbol_ref_na (_S(G95X_S_SYMBOL), h->symbol, &h->x.where_sy, 0);

      G95X_ADD_LST (_S(G95X_S_COMPONENT G95X_S_HN G95X_S_SPEC G95X_S_HN
		    G95X_S_LIST));
      for (c = h->value.constructor.c; c; c = c->next)
	{
	  if (c->x.dummy)
	    continue;
	  G95X_DUMP_SUBOBJ_AN (component_spec, c);
	}
      G95X_END_LST ();

      break;
    case EXPR_ARRAY:

      if (h->x.has_ts)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_TYPE G95X_S_HN G95X_S_SPEC), typespec,
			     &h->ts);

      G95X_ADD_LST (_S(G95X_S_ARRAY G95X_S_HN G95X_S_CONSTRUCTOR G95X_S_HN 
                    G95X_S_VALUE G95X_S_HN G95X_S_LIST));
      for (c = h->value.constructor.c; c; c = c->next)
	if (c->iterator)
	  G95X_DUMP_SUBOBJ_AN (constructor, c);
	else
	  G95X_DUMP_SUBOBJ_AN (expr, c->expr);
      G95X_END_LST ();

      break;

    case EXPR_SUBSTRING:
    case EXPR_CONSTANT:

      if (h->ts.type != BT_CHARACTER)
	dump_constant_location (h);

      if (h->x.boz)
	G95X_ADD_ATT_INT (_S(G95X_S_BOZ), 1);
      if ((h->ts.type == BT_CHARACTER) && h->value.character.hollerith)
	G95X_ADD_ATT_INT (_S(G95X_S_HOLLERITH), 1);
      G95X_ADD_ATT_CST (_S(G95X_S_VALUE), g95x_get_constant_as_string (h, 0));
      if (g95x_option.xml_defn)
        G95X_ADD_ATT_OBJ (_S(G95X_S_CONSTANT G95X_S_HN G95X_S_TYPE),
			  g95x_type_id (&h->ts));
      else
        G95X_ADD_ATT_STR (_S(G95X_S_CONSTANT G95X_S_HN G95X_S_TYPE),
  			  g95x_type_name (&h->ts));
      if (h->ts.x.kind)
	G95X_DUMP_SUBOBJ_NA (_S(G95X_S_KIND), expr, h->ts.x.kind->kind);

      if (h->ts.type == BT_CHARACTER)
	dump_constant_location (h);


      break;

    case EXPR_OP:

      G95X_EX(
        "!EXAMPLE:binary-op-expr\n"
        "      N = 2 + 3\n"
        "      END\n"
      );

      G95X_EX(
        "!EXAMPLE:unary-op-expr\n"
        "      N = + 3\n"
        "      END\n"
      );

      if (h->value.op.operator == INTRINSIC_PAREN)
	{
          G95X_EX(
            "!EXAMPLE:paren-expr\n"
            "      N = (N)\n"
            "      END\n"
          );

	  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_EXPR), expr, h->value.op.op1);
	}
      else if (h->value.op.operator == INTRINSIC_USER)
	{
	  g95x_die ("");
	}
      else
	{
	  dump_expr_op (h, h->value.op.op1, h->value.op.op2);
	}

      break;
    case EXPR_VARIABLE:
      h->x.where_sy = h->x.where;
      dump_global_symbol_ref_na (_S(G95X_S_SYMBOL), h->symbol, &h->x.where_sy, 0);
      break;

    case EXPR_NULL:
      h->x.where_sy = h->x.where;
      g95x_locate_name (&h->x.where_sy);

      if (g95x_option.xml_defn)
	G95X_ADD_ATT_OBJ (_S(G95X_S_FUNCTION), g95_find_function ("null"));

      dump_local_generic_ref_na (_S(G95X_S_GENERIC),
				 g95x_resolve_generic ("null"),
				 &h->x.where_sy);

      break;
    case EXPR_FUNCTION:
      if (h->x.is_operator)
	{

	  k = h->value.function.actual;
	  dump_expr_op (h, k->u.expr, k->next->u.expr);

	}
      else
	{

	  h->x.where_sy = h->x.where;
	  g95x_locate_name (&h->x.where_sy);
	  if (h->x.generic)
	    {			/* generic call */
	      dump_local_generic_ref_na (_S(G95X_S_GENERIC),
					 g95x_resolve_generic (h->value.
							       function.name),
					 &h->x.where_sy);
	      if (g95x_option.xml_defn)
		G95X_ADD_ATT_OBJ (_S(G95X_S_FUNCTION), h->symbol);
	    }
	  else if (h->symbol)
	    {			/* ordinary function call */
	      dump_global_symbol_ref_na (_S(G95X_S_FUNCTION), h->symbol,
					 &h->x.where_sy, 0);
	    }
	  else if (h->x.symbol)
	    {			/* intrinsic call */
	      dump_local_generic_ref_na (_S(G95X_S_GENERIC),
					 g95x_resolve_generic (h->value.
							       function.name),
					 &h->x.where_sy);
	      if (g95x_option.xml_defn)
		G95X_ADD_ATT_OBJ (_S(G95X_S_FUNCTION), h->value.function.isym);
	    }
	  else if (h->value.function.pointer)
	    {
	      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_FUNCTION), expr,
				   h->value.function.pointer);
	    }
	  else
	    {
	      g95x_die ("");
	    }

	  dump_actual_arglist (h->value.function.actual);

	}

      break;
    case EXPR_PROCEDURE:
      h->x.where_sy = h->x.where;
      dump_global_symbol_ref_na (_S(G95X_S_SYMBOL), h->symbol, &h->x.where_sy, 0);
      break;

    default:
      g95x_die ("Unknown expr type");
      break;
    }

  if (h->ref)
    {
      g95_typespec *ts;
      G95X_ADD_LST (_S(G95X_S_REFERENCE G95X_S_HN G95X_S_LIST));
      ts = &h->symbol->ts;
      for (r = h->ref; r; r = r->next) 
      {
        r->x.ts = ts;
        if (r->type == REF_COMPONENT)
          ts = &r->u.c.component->ts;
	G95X_DUMP_SUBOBJ_AN (ref, r);
      }
      G95X_END_LST ();
    }


  G95X_END_OBJ (expr, h);

  return;
}
#endif



void
G95X_DUMP_FCT (coshape_spec, g95x_coshape_spec * cs)
#ifdef INTF
;
#else
{
  int i = cs->i;
  g95_coarray_spec *cas = cs->cas;
  g95_expr *lower = cas->lower[i];
  g95_expr *upper = cas->upper[i];

  G95X_FULLIFY (lower);
  G95X_FULLIFY (upper);

  G95X_ADD_OBJ_NI (_S(G95X_S_CO G95X_S_HN G95X_S_SHAPE G95X_S_HN G95X_S_SPEC),
		   &cas->lower[i]);
  G95X_DUMP_ATTR_FCT (coshape_spec, cs);
  G95X_DUMP_WHERE (&cas->x.shape_spec_where[i]);


  if (g95x_valid_expr (lower))
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LOWER G95X_S_HN G95X_S_BOUND), expr, lower);

  if (upper)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_UPPER G95X_S_HN G95X_S_BOUND), expr, upper);
  else if ((cas->type == CAS_ASSUMED) && (i == (cas->corank - 1)) && (!upper))
    {
      dump_expr_chst_na (_S(G95X_S_UPPER G95X_S_HN G95X_S_BOUND),
			 &cas->x.assumed_where.lb2, &cas->x.assumed_where);
    }

  G95X_END_OBJ (coshape_spec, cs);
}
#endif



void
G95X_DUMP_FCT (coarray_spec, g95_coarray_spec * cas)
#ifdef INTF
;
#else
{
  int i;
  g95x_coshape_spec cs;

  G95X_ADD_OBJ_NI (_S(G95X_S_CO G95X_S_HN G95X_S_ARRAY G95X_S_HN G95X_S_SPEC),
                   cas);

  G95X_DUMP_ATTR_FCT (coarray_spec, cas);
  G95X_DUMP_WHERE (&cas->x.where);
  G95X_ADD_ATT_STR (_S(G95X_S_CO G95X_S_HN G95X_S_SHAPE),
		    g95x_coarray_spec_type (cas));

  G95X_ADD_LST (_S(G95X_S_CO G95X_S_HN G95X_S_SHAPE G95X_S_HN G95X_S_SPEC
		G95X_S_HN G95X_S_LIST));

  for (i = 0; i < cas->corank; i++)
    {
      cs.i = i;
      cs.cas = cas;
      if (i < cas->corank-1)
        {
          g95_expr *lower = cas->lower[i], 
                   *upper = cas->upper[i];
          cs.where = upper->x.where;
          if (g95x_valid_expr (lower))
            {
              cs.where.lb1 = lower->x.where.lb1;
              cs.where.column1 = lower->x.where.column1;
            }
        }
      else
        {
          cs.where = cas->x.assumed_where;
        }
      G95X_DUMP_SUBOBJ_AN (coshape_spec, &cs);
    }
  G95X_END_LST ();

  G95X_END_OBJ (coarray_spec, cas);

}
#endif


void
G95X_DUMP_FCT (co_alloc_spec, g95x_co_alloc_spec * ss)
#ifdef INTF
;
#else
{
  g95_expr* lb;
  g95_expr* ub;

  lb = ss->lb;
  ub = ss->ub;

  G95X_ADD_OBJ_NI (_S(G95X_S_CO G95X_S_HN G95X_S_ALLOCATE G95X_S_HN G95X_S_SPEC), ss);
  G95X_DUMP_ATTR_FCT (co_alloc_spec, ss);

  if (lb)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LOWER G95X_S_HN G95X_S_BOUND), expr, lb);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_UPPER G95X_S_HN G95X_S_BOUND), expr, ub);
  
  G95X_END_OBJ (co_alloc_spec, ss);

}
#endif

void
G95X_DUMP_FCT (alloc_spec, g95x_alloc_spec * ss)
#ifdef INTF
;
#else
{
  g95_expr* lb;
  g95_expr* ub;

  lb = ss->lb;
  ub = ss->ub;

  G95X_ADD_OBJ_NI (_S(G95X_S_ALLOCATE G95X_S_HN G95X_S_SPEC), ss);
  G95X_DUMP_ATTR_FCT (alloc_spec, ss);

  if (lb)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LOWER G95X_S_HN G95X_S_BOUND), expr, lb);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_UPPER G95X_S_HN G95X_S_BOUND), expr, ub);
  
  G95X_END_OBJ (alloc_spec, ss);
}
#endif


void
G95X_DUMP_FCT (shape_spec, g95x_shape_spec * ss)
#ifdef INTF
;
#else
{
  int i = ss->i;
  g95_array_spec *as = ss->as;
  g95_expr *lower = as->lower[i];
  g95_expr *upper = as->upper[i];

  G95X_FULLIFY (lower);
  G95X_FULLIFY (upper);

  G95X_ADD_OBJ_NI (_S(G95X_S_SHAPE G95X_S_HN G95X_S_SPEC), &as->lower[i]);
  G95X_DUMP_ATTR_FCT (shape_spec, ss);
  G95X_DUMP_WHERE (&as->x.shape_spec_where[i]);

  if (g95x_valid_expr (lower))
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LOWER G95X_S_HN G95X_S_BOUND), expr, lower);

  if (g95x_valid_expr (upper))
    {
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_UPPER G95X_S_HN G95X_S_BOUND), expr, upper);
    }
  else if ((as->type == AS_ASSUMED_SIZE) && (i == (as->rank - 1)) && (!upper))
    {
      dump_expr_chst_na (_S(G95X_S_UPPER G95X_S_HN G95X_S_BOUND),
			 &as->x.assumed_where.lb2, &as->x.assumed_where);
    }

  G95X_END_OBJ (shape_spec, ss);
}
#endif


void
G95X_DUMP_FCT (array_spec, g95_array_spec * as)
#ifdef INTF
;
#else
{
  int i;
  g95x_shape_spec ss;

  G95X_ADD_OBJ_NI (_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC), as);

  G95X_ADD_ATT_STR (_S(G95X_S_SHAPE), g95x_array_spec_type (as));
  G95X_DUMP_ATTR_FCT (array_spec, as);

  G95X_DUMP_WHERE (&as->x.where);

  G95X_ADD_LST (_S(G95X_S_SHAPE G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
  for (i = 0; i < as->rank; i++)
    {
      ss.as = as;
      ss.i = i;
      G95X_DUMP_SUBOBJ_AN (shape_spec, &ss);
    }
  G95X_END_LST ();

  G95X_END_OBJ (array_spec, as);

}
#endif


void
G95X_DUMP_FCT (component_decl, g95x_voidp_list * f)
#ifdef INTF
;
#else
{
  void *psym;
  g95_expr *len = NULL, *kind = NULL, *val = NULL;
  g95_array_spec *as = NULL;
  g95_coarray_spec *cas = NULL;
  g95_charlen *cl = NULL;
  g95_typespec *ts = NULL;
  const void *typeid;

  G95X_ADD_OBJ_NI (_S(G95X_S_COMPONENT G95X_S_HN G95X_S_DECL), f);
  G95X_DUMP_ATTR_FCT (component_decl, f);
  if (!f->in_derived || !(f->type == G95X_VOIDP_COMPNT))
    g95x_die ("");
  G95X_DUMP_WHERE (&f->where);

  g95_component *comp = f->u.component;
  psym = comp;
  g95x_get_c_lkacv (comp, &len, &kind, &as, &cas, &val);
  ts = &comp->ts;
  get_tklc (ts, &typeid, &kind, &len, &cl);


  f->where_obj = f->where;
  g95x_locate_name (&f->where_obj);
  dump_local_symbol_ref_na (_S(G95X_S_OBJECT G95X_S_HN G95X_S_NAME), psym,
			    &f->where_obj);

  if (f->dimension)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC), array_spec, as);
  if (cas)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CO G95X_S_HN G95X_S_ARRAY G95X_S_HN
			 G95X_S_SPEC), coarray_spec, cas);
  if (f->length)
    {
      if (len)
	{
	  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CHAR G95X_S_HN G95X_S_LENGTH), expr,
			       len);
	}
      else if (cl && (cl->length == NULL))
	{
	  dump_expr_chst_na (_S(G95X_S_CHAR G95X_S_HN G95X_S_LENGTH), cl,
			     &cl->x.chst_where);
	}
    }
  if (val)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_INITIALIZATION G95X_S_HN G95X_S_EXPR), expr,
			 val);


  G95X_END_OBJ (component_decl, f);
}
#endif


void
G95X_DUMP_FCT (entity_decl, g95x_voidp_list * f)
#ifdef INTF
;
#else
{
  void *psym;
  g95_expr *len = NULL, *kind = NULL, *val = NULL;
  g95_array_spec *as = NULL;
  g95_coarray_spec *cas = NULL;
  g95_charlen *cl = NULL;
  g95_typespec *ts = NULL;
  const void *typeid;

  G95X_ADD_OBJ_NI (_S(G95X_S_ENTITY G95X_S_HN G95X_S_DECL), f);
  G95X_DUMP_ATTR_FCT (entity_decl, f);
  if ((f->in_derived) || (f->type == G95X_VOIDP_COMPNT))
    g95x_die ("");
  G95X_DUMP_WHERE (&f->where);

  if (f->type == G95X_VOIDP_SYMBOL)
    {
      g95_symbol *sym = f->u.symbol;
      if (g95x_symbol_intrinsic (sym))
	{
	  psym = sym->ts.type != BT_PROCEDURE ?
	    g95_find_function (sym->name) : g95_find_subroutine (sym->name);
	}
      else
	{
	  psym = sym;
	  g95x_get_s_lkacv (sym, &len, &kind, &as, &cas, &val);
	  ts = &sym->ts;
	  get_tklc (ts, &typeid, &kind, &len, &cl);
	}
    }
  else
    {
      psym = f->u.voidp;
    }


  f->where_obj = f->where;
  g95x_locate_name (&f->where_obj);
  dump_local_symbol_ref_na (_S(G95X_S_OBJECT G95X_S_HN G95X_S_NAME), psym,
			    &f->where_obj);


  if (f->dimension)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC), array_spec, as);
  if (cas)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CO G95X_S_HN G95X_S_ARRAY G95X_S_HN
			 G95X_S_SPEC), coarray_spec, cas);
  if (f->length)
    {
      if (len)
	{
	  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CHAR G95X_S_HN G95X_S_LENGTH), expr,
			       len);
	}
      else if (cl && (cl->length == NULL))
	{
	  dump_expr_chst_na (_S(G95X_S_CHAR G95X_S_HN G95X_S_LENGTH), cl,
			     &cl->x.chst_where);
	}
    }
  if (f->init)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_INITIALIZATION G95X_S_HN G95X_S_EXPR), expr,
			 val);

  G95X_END_OBJ (entity_decl, f);
}
#endif




void
G95X_DUMP_FCT (expr_chst, g95x_expr_chst * ch)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_T_NI (_S(G95X_S_EXPR), _S(G95X_S_STAR), ch->addr);

  G95X_DUMP_ATTR_FCT (expr_chst, ch);
  G95X_DUMP_WHERE (ch->where);
  G95X_END_OBJ (expr_chst, ch);
}
#endif

void
G95X_DUMP_FCT (enumeration, g95x_voidp_list * sl)
#ifdef INTF
;
#else
{
  g95_symbol *sym = sl->u.symbol;
  g95_expr *init;

  G95X_ADD_OBJ_NI (_S(G95X_S_ENUMERATION), sl);
  G95X_DUMP_ATTR_FCT (enumeration, sl);
  G95X_DUMP_WHERE (&sl->where);

  sl->where_obj = sl->where;
  g95x_locate_name (&sl->where_obj);
  dump_local_symbol_ref_na (_S(G95X_S_OBJECT G95X_S_HN G95X_S_NAME), sym,
			    &sl->where_obj);


  init = sym->value;
  G95X_FULLIFY (init);

  if (g95x_valid_expr (init))
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_INITIALIZATION G95X_S_HN G95X_S_EXPR), expr,
			 init);

  G95X_END_OBJ (enumeration, sl);
}
#endif

void
G95X_DUMP_FCT (namelist_group, g95x_voidp_list * sl)
#ifdef INTF
;
#else
{
  g95x_voidp_list *sl1;

  if (sl->u.symbol->attr.flavor != FL_NAMELIST)
    g95x_die ("");

  G95X_ADD_OBJ_NI (_S(G95X_S_NAMELIST G95X_S_HN G95X_S_GROUP), sl);
  G95X_DUMP_ATTR_FCT (namelist_group, sl);

  dump_local_symbol_ref_na (_S(G95X_S_NAMELIST G95X_S_HN G95X_S_GROUP G95X_S_HN
			    G95X_S_NAME), sl->u.symbol, &sl->where);

  G95X_ADD_LST (_S(G95X_S_NAMELIST G95X_S_HN G95X_S_GROUP G95X_S_HN G95X_S_OBJECT
		G95X_S_HN G95X_S_LIST));

  for (sl1 = sl->prev; sl1; sl1 = sl1->prev)
    {

      if (sl1->u.symbol->attr.flavor == FL_NAMELIST)
	break;

      dump_local_symbol_ref_an (sl1->u.symbol, &sl1->where);

    }
  G95X_END_LST ();
  G95X_END_OBJ (namelist_group, sl);
}
#endif

void
G95X_DUMP_FCT (data_statement_set, g95_data * d)
#ifdef INTF
;
#else
{
  g95_data_variable *var;
  g95_data_value *val;

  G95X_ADD_OBJ_NI (_S(G95X_S_DATA G95X_S_HN G95X_S_STATEMENT G95X_S_HN
		   G95X_S_SET), d);

  G95X_ADD_LST (_S(G95X_S_DATA G95X_S_HN G95X_S_STATEMENT G95X_S_HN G95X_S_OBJECT
		G95X_S_HN G95X_S_LIST));
  for (var = d->var; var; var = var->next)
    if (var->expr)
      G95X_DUMP_SUBOBJ_AN (expr, var->expr);
    else
      G95X_DUMP_SUBOBJ_AN (data_var, var);
  G95X_END_LST ();

  G95X_ADD_LST (_S(G95X_S_DATA G95X_S_HN G95X_S_STATEMENT G95X_S_HN G95X_S_VALUE
		G95X_S_HN G95X_S_LIST));
  for (val = d->value; val; val = val->next)
    G95X_DUMP_SUBOBJ_AN (data_val, val);
  G95X_END_LST ();

  G95X_END_OBJ (data_statement_set, d);

}
#endif

void
G95X_DUMP_FCT (alt_return, g95x_alt_return * ar)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_NI (_S(G95X_S_ALT G95X_S_HN G95X_S_RETURN), ar->addr);
  G95X_DUMP_ATTR_FCT (alt_return, ar);
  G95X_DUMP_WHERE (ar->where);

  G95X_END_OBJ (alt_return, ar);
}
#endif

void
G95X_DUMP_FCT (formal_arg, g95_formal_arglist * fa)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_T_NI (_S(G95X_S_DUMMY G95X_S_HN G95X_S_ARG), 
                     g95x_formal_arg_type (fa), fa);

  G95X_DUMP_ATTR_FCT (formal_arg, fa);

/* TODO : we have to handle the %VAL( X ) case too */


  if (fa->sym)
    {
      dump_local_symbol_ref_na (_S(G95X_S_DUMMY G95X_S_HN G95X_S_ARG G95X_S_HN
				G95X_S_NAME), fa->sym, &fa->x.where);
    }
  else
    {
      g95x_alt_return ar;
      ar.addr = &fa->cb;
      ar.where = &fa->x.where;
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ALT G95X_S_HN G95X_S_RETURN), alt_return,
			   &ar);
    }

  G95X_END_OBJ (formal_arg, fa);
}
#endif

void
G95X_DUMP_FCT (data_var, g95_data_variable * var)
#ifdef INTF
;
#else
{
  g95_data_variable *v;

  G95X_ADD_OBJ_NI (_S(G95X_S_DATA G95X_S_HN G95X_S_STATEMENT G95X_S_HN
		   G95X_S_OBJECT), var);
  G95X_DUMP_ATTR_FCT (data_var, var);

  if (var->expr)
    g95x_die ("");
  else if (var->list)
    {
      G95X_ADD_LST (_S(G95X_S_DATA G95X_S_HN G95X_S_DO G95X_S_HN G95X_S_OBJECT
		    G95X_S_HN G95X_S_LIST));
      for (v = var->list; v; v = v->next)
	if (v->expr)
	  G95X_DUMP_SUBOBJ_AN (expr, v->expr);
	else
	  G95X_DUMP_SUBOBJ_AN (data_var, v);
      G95X_END_LST ();
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_DATA G95X_S_HN G95X_S_IMPLIED G95X_S_HN
			   G95X_S_DO G95X_S_HN G95X_S_ITERATOR), iterator,
			   &var->iter);
    }

  G95X_END_OBJ (data_var, var);
}
#endif

void
G95X_DUMP_FCT (data_val, g95_data_value * val)
#ifdef INTF
;
#else
{
  g95_expr * v = val->expr, *r = val->x.repeat;
  G95X_FULLIFY (v);
  G95X_FULLIFY (r);

  G95X_ADD_OBJ_NI (_S(G95X_S_DATA G95X_S_HN G95X_S_STATEMENT G95X_S_HN
		   G95X_S_VALUE), val);

  if (r)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_DATA G95X_S_HN G95X_S_STATEMENT G95X_S_HN
			 G95X_S_REPEAT), expr, r);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_DATA G95X_S_HN G95X_S_STATEMENT G95X_S_HN
		       G95X_S_CONSTANT), expr, v);

  G95X_END_OBJ (data_val, val);

}
#endif

void
G95X_DUMP_FCT (common_block_object, g95x_voidp_list * sl)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_NI (_S(G95X_S_COMMON G95X_S_HN G95X_S_BLOCK G95X_S_HN
		   G95X_S_OBJECT), sl);
  G95X_DUMP_ATTR_FCT (common_block_object, sl);

#ifdef G95X_DUMP_ATTR
  sl->where_obj = sl->where;
  g95x_locate_name (&sl->where_obj);
#endif

  G95X_DUMP_WHERE (&sl->where);

  dump_local_symbol_ref_na (_S(G95X_S_OBJECT G95X_S_HN G95X_S_NAME), sl->u.symbol,
			    &sl->where_obj);


  if (sl->dimension)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC), array_spec,
			 sl->u.symbol->as);

  G95X_END_OBJ (common_block_object, sl);
}
#endif


void
G95X_DUMP_FCT (common_block_decl, g95x_voidp_list * sl)
#ifdef INTF
;
#else
{
  g95x_voidp_list *sl1;

  if (sl->type != G95X_VOIDP_COMMON)
    g95x_die ("");

  G95X_ADD_OBJ_NI (_S(G95X_S_COMMON G95X_S_HN G95X_S_BLOCK G95X_S_HN G95X_S_DECL),
		   sl);
  G95X_DUMP_ATTR_FCT (common_block_decl, sl);

  if (sl->u.common == (struct g95_common_head *) g95x_dump_ns->blank_common)
    {
      if (g95x_option.xml_defn)
	G95X_ADD_ATT_OBJ (_S(G95X_S_COMMON G95X_S_HN G95X_S_BLOCK G95X_S_HN
			  G95X_S_NAME), sl->u.common);
    }
  else
    {
      dump_local_common_ref_na (_S(G95X_S_COMMON G95X_S_HN G95X_S_BLOCK G95X_S_HN
				G95X_S_NAME), sl->u.common, &sl->where);
    }

  G95X_ADD_LST (_S(G95X_S_COMMON G95X_S_HN G95X_S_BLOCK G95X_S_HN G95X_S_OBJECT
		G95X_S_HN G95X_S_LIST));
  for (sl1 = sl->prev; sl1; sl1 = sl1->prev)
    {
      if (sl1->type != G95X_VOIDP_SYMBOL)
	break;
      G95X_DUMP_SUBOBJ_AN (common_block_object, sl1);
    }
  G95X_END_LST ();
  G95X_END_OBJ (common_block_decl, sl);
}
#endif


void
G95X_DUMP_FCT (named_constant_def, g95x_voidp_list * sl)
#ifdef INTF
;
#else
{
  g95_expr *e;
  e = sl->u.symbol->value;
  G95X_FULLIFY (e);
  G95X_ADD_OBJ_NI (_S(G95X_S_NAMED G95X_S_HN G95X_S_CONSTANT G95X_S_HN
		   G95X_S_DEF), sl);
  G95X_DUMP_ATTR_FCT (named_constant_def, sl);
  G95X_DUMP_WHERE (&sl->where);

  sl->where_obj = sl->where;
  g95x_locate_name (&sl->where_obj);
  dump_local_symbol_ref_na (_S(G95X_S_NAMED G95X_S_HN G95X_S_CONSTANT),
			    sl->u.symbol, &sl->where_obj);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_INITIALIZATION G95X_S_HN G95X_S_EXPR), expr, e);
  G95X_END_OBJ (named_constant_def, sl);
}
#endif



void
G95X_DUMP_FCT (statement, g95x_statement * sta)
#ifdef INTF
;
#else
{

  G95X_ADD_OBJ_T_NI (_S(G95X_S_STATEMENT), 
                     g95x_statement_type (sta), sta);

  G95X_DUMP_ATTR_FCT (statement, sta);

  G95X_DUMP_WHERE (&sta->where);


  if (sta->here)
    dump_label_na (_S(G95X_S_STATEMENT G95X_S_HN G95X_S_LABEL), &sta->here,
		   &sta->here->x.where);


  if (sta->implied_enddo && g95x_option.xml_defn)
    G95X_ADD_ATT_INT (_S(G95X_S_ENDDO), sta->implied_enddo);

  switch (sta->type)
    {
    case ST_DO:
      dump_statement_DO (sta);
      break;
    case ST_IF_BLOCK:
      dump_statement_IF_BLOCK (sta);
      break;
    case ST_SELECT_CASE:
      dump_statement_SELECT_CASE (sta);
      break;
    case ST_WHERE_BLOCK:
      dump_statement_WHERE_BLOCK (sta);
      break;
    case ST_FORALL_BLOCK:
      dump_statement_FORALL_BLOCK (sta);
      break;
    default:
      break;
    }

  dump_statement_code (sta);

  switch (sta->type)
    {
    case ST_BLOCK_DATA:
      dump_statement_BLOCK_DATA (sta);
      break;
    case ST_END_BLOCK_DATA:
      dump_statement_END_BLOCK_DATA (sta);
      break;
    case ST_PROGRAM:
      dump_statement_PROGRAM (sta);
      break;
    case ST_END_PROGRAM:
      dump_statement_END_PROGRAM (sta);
      break;
    case ST_ATTR_DECL:
      dump_statement_ATTR_DECL (sta);
      break;
    case ST_INTERFACE:
      dump_statement_INTERFACE (sta);
      break;
    case ST_END_INTERFACE:
      dump_statement_END_INTERFACE (sta);
      break;
    case ST_FORMAT:
      dump_statement_FORMAT (sta);
      break;
    case ST_ENUM:
      dump_statement_ENUM (sta);
      break;
    case ST_IMPLICIT:
      dump_statement_IMPLICIT (sta);
      break;
    case ST_PARAMETER:
      dump_statement_PARAMETER (sta);
      break;
    case ST_COMMON:
      dump_statement_COMMON (sta);
      break;
    case ST_DERIVED_DECL:
      dump_statement_DERIVED_DECL (sta);
      break;
    case ST_END_TYPE:
      dump_statement_END_TYPE (sta);
      break;
    case ST_DATA:
      dump_statement_DATA (sta);
      break;
    case ST_EQUIVALENCE:
      dump_statement_EQUIVALENCE (sta);
      break;
    case ST_NAMELIST:
      dump_statement_NAMELIST (sta);
      break;
    case ST_DATA_DECL:
      dump_statement_DATA_DECL (sta);
      break;
    case ST_ENUMERATOR:
      dump_statement_ENUMERATOR (sta);
      break;
    case ST_SUBROUTINE:
      dump_statement_SUBROUTINE (sta);
      break;
    case ST_END_SUBROUTINE:
      dump_statement_END_SUBROUTINE (sta);
      break;
    case ST_MODULE:
      dump_statement_MODULE (sta);
      break;
    case ST_END_MODULE:
      dump_statement_END_MODULE (sta);
      break;
    case ST_ENTRY:
      dump_statement_ENTRY (sta);
      break;
    case ST_FUNCTION:
      dump_statement_FUNCTION (sta);
      break;
    case ST_END_FUNCTION:
      dump_statement_END_FUNCTION (sta);
      break;
    case ST_STATEMENT_FUNCTION:
      dump_statement_STATEMENT_FUNCTION (sta);
      break;
    case ST_ELSE:
      dump_statement_ELSE (sta);
      break;
    case ST_ENDIF:
      dump_statement_ENDIF (sta);
      break;
    case ST_CYCLE:
      dump_statement_CYCLE (sta);
      break;
    case ST_EXIT:
      dump_statement_EXIT (sta);
      break;
    case ST_ENDDO:
      dump_statement_ENDDO (sta);
      break;
    case ST_CASE:
      dump_statement_CASE (sta);
      break;
    case ST_END_SELECT:
      dump_statement_END_SELECT (sta);
      break;
    case ST_ELSEWHERE:
      dump_statement_ELSEWHERE (sta);
      break;
    case ST_END_WHERE:
      dump_statement_END_WHERE (sta);
      break;
    case ST_END_FORALL:
      dump_statement_END_FORALL (sta);
      break;
    case ST_USE:
      dump_statement_USE (sta);
      break;
    case ST_SIMPLE_IF:
      dump_statement_SIMPLE_IF (sta);
      break;
    case ST_WHERE:
      dump_statement_WHERE (sta);
      break;
    case ST_FORALL:
      dump_statement_FORALL (sta);
      break;
    case ST_MODULE_PROC:
      dump_statement_MODULE_PROC (sta);
      break;
    case ST_IMPORT:
      dump_statement_IMPORT (sta);
      break;
    default:
      break;
    }
  switch (sta->type)
    {
    case ST_CALL:
      G95X_EX(
        "!EXAMPLE:call-with-arg-lst-stmt\n"
        "      CALL SUBNAME( X, Y )\n"
        "      END\n"
      );
      G95X_EX(
        "!EXAMPLE:call-without-arg-lst-stmt\n"
        "      CALL SUBNAME\n"
        "      END\n"
      );
      break;
    case ST_BLOCK_DATA:
      G95X_EX(
        "!EXAMPLE:anonymous-block-data-stmt\n"
        "      BLOCKDATA\n"
        "      END\n"
      );
      G95X_EX(
        "!EXAMPLE:named-block-data-stmt\n"
        "      BLOCKDATA BLOCKNAME\n"
        "      END\n"
      );
      break;
    case ST_END_BLOCK_DATA:
      G95X_EX(
        "!EXAMPLE:anonymous-end-block-data-stmt\n"
        "      BLOCKDATA\n"
        "      END\n"
      );
      G95X_EX(
        "!EXAMPLE:end-block-data-stmt\n"
        "      BLOCKDATA\n"
        "      END BLOCKDATA\n"
      );
      G95X_EX(
        "!EXAMPLE:named-end-block-data-stmt\n"
        "      BLOCKDATA BLOCKNAME\n"
        "      END BLOCKDATA BLOCKNAME\n"
      );
      break;
    case ST_IF_BLOCK:
      G95X_EX(
        "!EXAMPLE:anonymous-if-then-stmt\n"
        "      IF( .TRUE. ) THEN\n"
        "      ENDIF\n"
        "      END\n"
      );
      G95X_EX(
        "!EXAMPLE:named-if-then-stmt\n"
        "      MYIF : IF( .TRUE. ) THEN\n"
        "      ENDIF MYIF\n"
        "      END\n"
      );
      break;
    case ST_ENDIF:
      G95X_EX(
        "!EXAMPLE:anonymous-end-if-stmt\n"
        "      IF( .TRUE. ) THEN\n"
        "      ENDIF\n"
        "      END\n"
      );
      G95X_EX(
        "!EXAMPLE:named-end-if--stmt\n"
        "      MYIF : IF( .TRUE. ) THEN\n"
        "      ENDIF MYIF\n"
        "      END\n"
      );
      break;
    default:
      break;
    }

  G95X_END_OBJ (statement, sta);


}
#endif


void
G95X_DUMP_FCT (component, g95_component * c)
#ifdef INTF
;
#else
{
  g95_array_spec *as;
  g95_coarray_spec *cas;
  g95_expr *length, *kind, *value;
  const void *typeid;

  G95X_ADD_OBJ_NI (_S(G95X_S_COMPONENT), c);
  G95X_DUMP_ATTR_FCT (component, c);

  g95x_get_c_lkacv (c, &length, &kind, &as, &cas, &value);

  G95X_ADD_ATT_STR (_S(G95X_S_NAME), c->name);

  if (g95x_option.xml_defn)
    {

      if (as)
	G95X_ADD_ATT_OBJ (_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC), as);

      if (cas)
	G95X_ADD_ATT_OBJ (_S(G95X_S_CO G95X_S_HN G95X_S_ARRAY G95X_S_HN
			  G95X_S_SPEC), cas);

      if (length)
	G95X_ADD_ATT_OBJ (_S(G95X_S_LENGTH), length);

      if (kind)
	G95X_ADD_ATT_OBJ (_S(G95X_S_KIND), kind);

      if (value)
	G95X_ADD_ATT_OBJ (_S(G95X_S_VALUE), value);

      if (c->pointer)
	G95X_ADD_ATT_INT (_S(G95X_S_POINTER), 1);

      if (c->allocatable)
	G95X_ADD_ATT_INT (_S(G95X_S_ALLOCATABLE), 1);

      if (c->nopass)
	G95X_ADD_ATT_INT (_S(G95X_S_NOPASS), 1);

      typeid = g95x_type_id (&c->ts);
      if (typeid)
	G95X_ADD_ATT_OBJ (_S(G95X_S_TYPE), typeid);

    }

  G95X_END_OBJ (component, c);
}
#endif

#ifndef INTF

static void
dump_symbol_attributes (g95_symbol * sym)
{
  g95_namelist *nam;
  g95_array_spec *as;
  g95_coarray_spec *cas;
  g95_expr *value, *length, *kind;
  g95_charlen *cl;
  g95_formal_arglist *fa;
  const void *typeid;

//G95X_ADD_ATT_STR (_S(G95X_S_FLAVOR), g95x_flavor_type (sym));

  if (g95x_option.xml_defn)
    {
      g95x_get_s_acv (sym, &as, &cas, &value);
      get_tklc (&sym->ts, &typeid, &kind, &length, &cl);

      print_attributes_list (&sym->attr);

      if (value)
	G95X_ADD_ATT_OBJ (_S(G95X_S_VALUE), value);

      if (as)
	G95X_ADD_ATT_OBJ (_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC), as);

      if (cas)
	G95X_ADD_ATT_OBJ (_S(G95X_S_CO G95X_S_HN G95X_S_ARRAY G95X_S_HN
			  G95X_S_SPEC), cas);

      if (kind)
	G95X_ADD_ATT_OBJ (_S(G95X_S_KIND), kind);

      if (length)
	G95X_ADD_ATT_OBJ (_S(G95X_S_LENGTH), length);
      else if (cl && (cl->length == NULL))
	G95X_ADD_ATT_OBJ (_S(G95X_S_LENGTH), cl);

      if (typeid)
	G95X_ADD_ATT_OBJ (_S(G95X_S_TYPE), typeid);

      if (sym->ts.interface)
	G95X_ADD_ATT_OBJ (_S(G95X_S_INTERFACE), sym->ts.interface);

      if (sym->attr.bind)
	{
	  g95_expr *bind = sym->x.bind;
	  G95X_ADD_ATT_STR (_S(G95X_S_BIND), _S(G95X_S_C));
	  G95X_FULLIFY (bind);
	  if (bind && g95x_valid_expr (bind))
	    G95X_ADD_ATT_OBJ (_S(G95X_S_BIND G95X_S_HN G95X_S_NAME), bind);
	}

      switch (sym->attr.flavor)
	{
	case FL_NAMELIST:
	  G95X_ADD_LST (_S(G95X_S_SYMBOL G95X_S_HN G95X_S_LIST));
	  for (nam = sym->namelist; nam; nam = nam->next)
	    G95X_PSH_LST_OBJ (nam->sym);
	  G95X_END_LST ();
	  break;
	case FL_PROCEDURE:
	  G95X_ADD_LST (_S(G95X_S_DUMMY G95X_S_HN G95X_S_ARG G95X_S_HN
	        	G95X_S_NAME G95X_S_HN G95X_S_LIST));
	  for (fa = sym->formal; fa; fa = fa->next)
	    G95X_PSH_LST_OBJ (fa);
	  G95X_END_LST ();
	  break;
	default:
	  break;
	}

    }

}

#endif


void
G95X_DUMP_FCT (abstract, g95_symtree * st)
#ifdef INTF
;
#else
{
  g95_symbol *sym = st->n.sym;
  int glbl = g95x_symbol_global (sym);
  int dfnd = g95x_symbol_defined (sym);

//G95X_ADD_OBJ (_S(G95X_S_ABSTRACT G95X_S_HN G95X_S_INTERFACE), sym);
//G95X_ADD_OBJ_T_NI (_S(G95X_S_ABSTRACT G95X_S_HN G95X_S_INTERFACE), g95x_flavor_type (sym), sym);

  G95X_ADD_OBJ_T_NI (g95x_flavor_type (sym), _S(G95X_S_ABSTRACT G95X_S_HN G95X_S_INTERFACE), sym);
  G95X_DUMP_ATTR_FCT (abstract, st);

  if (glbl)
    G95X_ADD_ATT_STR (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_NAME),
		      g95x_symbol_gname (sym));
  if (dfnd)
    G95X_ADD_ATT_STR (_S(G95X_S_LOCAL G95X_S_HN G95X_S_NAME), st->name);

  dump_symbol_attributes (sym);

  G95X_END_OBJ (abstract, st);
}
#endif

void
G95X_DUMP_FCT (user_type, g95_symtree * st)
#ifdef INTF
;
#else
{
  g95_component *comp;
  g95_symbol *sym = st->n.sym;
  int glbl = g95x_symbol_global (sym);
  int dfnd = g95x_symbol_defined (sym);

  G95X_ADD_OBJ_T (_S(G95X_S_USER G95X_S_HN G95X_S_DEFINED G95X_S_HN G95X_S_TYPE),
  		  _S(G95X_S_DERIVED), sym);
  G95X_DUMP_ATTR_FCT (user_type, st);

  if (glbl)
    G95X_ADD_ATT_STR (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_NAME),
		      g95x_symbol_gname (sym));
  if (dfnd)
    G95X_ADD_ATT_STR (_S(G95X_S_LOCAL G95X_S_HN G95X_S_NAME), st->name);

  if (sym->attr.access == ACCESS_PUBLIC)
    G95X_ADD_ATT_STR (_S(G95X_S_ACCESS), _S(G95X_S_PUBLIC));
  G95X_ADD_LST (_S(G95X_S_COMPONENT G95X_S_HN G95X_S_LIST));
  for (comp = sym->components; comp; comp = comp->next)
    G95X_DUMP_SUBOBJ_AN (component, comp);
  G95X_END_LST ();


  G95X_END_OBJ (user_type, st);
}
#endif


void
G95X_DUMP_FCT (imported_component, g95_component * c)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_NI (_S(G95X_S_COMPONENT), c);
  G95X_DUMP_ATTR_FCT (imported_component, c);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), c->name);
  G95X_END_OBJ (imported_component, c);
}
#endif

void
G95X_DUMP_FCT (imported_abstract, g95_symtree * st)
#ifdef INTF
;
#else
{
  g95_symbol *sym = st->n.sym;
  G95X_ADD_OBJ (_S(G95X_S_ABSTRACT G95X_S_HN G95X_S_INTERFACE), sym);
  G95X_DUMP_ATTR_FCT (imported_abstract, st);
  G95X_ADD_ATT_STR (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_NAME),
		    g95x_symbol_gname (sym));
  G95X_END_OBJ (imported_abstract, st);
}
#endif

void
G95X_DUMP_FCT (imported_user_type, g95_symtree * st)
#ifdef INTF
;
#else
{
  g95_component *comp;
  g95_symbol *sym = st->n.sym;
  G95X_ADD_OBJ (_S(G95X_S_USER G95X_S_HN G95X_S_DEFINED G95X_S_HN G95X_S_TYPE),
		    sym);
  G95X_DUMP_ATTR_FCT (imported_user_type, st);
  G95X_ADD_ATT_STR (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_NAME),
		    g95x_symbol_gname (sym));
  G95X_ADD_LST (_S(G95X_S_COMPONENT G95X_S_HN G95X_S_LIST));
  for (comp = sym->components; comp; comp = comp->next)
    G95X_DUMP_SUBOBJ_AN (imported_component, comp);
  G95X_END_LST ();
  G95X_END_OBJ (imported_component, st);
}
#endif

void
G95X_DUMP_FCT (imported_symbol, g95_symtree * st)
#ifdef INTF
;
#else
{
  g95_symbol *sym = st->n.sym;
  G95X_ADD_OBJ_T_NI (g95x_flavor_type (sym), _S(G95X_S_GLOBAL), sym);

//G95X_ADD_OBJ_T_NI (_S(G95X_S_GLOBAL), g95x_flavor_type (sym), sym);
//G95X_ADD_OBJ (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_SYMBOL), sym);
//G95X_ADD_ATT_STR (_S(G95X_S_FLAVOR), g95x_flavor_type (sym));

  G95X_DUMP_ATTR_FCT (imported_symbol, st);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), g95x_symbol_gname (sym));
  G95X_END_OBJ (imported_symbol, st);
}
#endif

void
G95X_DUMP_FCT (association, g95_symtree * st)
#ifdef INTF
;
#else
{
  G95X_ADD_ALS (_S(G95X_S_LOCAL G95X_S_HN G95X_S_GLOBAL), st->name, st->n.sym);
  G95X_DUMP_ATTR_FCT (association, st);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), st->name);
  G95X_ADD_ATT_STR (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_NAME), g95x_symbol_gname (st->n.sym) );
  G95X_END_OBJ (association, st);
}
#endif

void
G95X_DUMP_FCT (symbol, g95_symtree * st)
#ifdef INTF
;
#else
{
  g95_symbol *sym = st->n.sym;
  int glbl = g95x_symbol_global (sym);
  int dfnd = g95x_symbol_defined (sym);

//G95X_ADD_OBJ (_S(G95X_S_SYMBOL), sym);

  if (glbl)
    G95X_ADD_OBJ_T_NI (g95x_flavor_type (sym), _S(G95X_S_GLOBAL), sym);
//  G95X_ADD_OBJ_T_NI (_S(G95X_S_GLOBAL), g95x_flavor_type (sym), sym);
  else
    G95X_ADD_OBJ_T_NI (g95x_flavor_type (sym), _S(G95X_S_LOCAL), sym);
//  G95X_ADD_OBJ_T_NI (_S(G95X_S_LOCAL), g95x_flavor_type (sym), sym);

  G95X_DUMP_ATTR_FCT (symbol, st);

  if (strcmp (BLANK_BLOCK_DATA_NAME, sym->name))
    {
      if (glbl)
        {
          G95X_ADD_ATT_STR (_S(G95X_S_NAME),
          		  g95x_symbol_gname (sym));
          if (dfnd)
            G95X_ADD_ATT_INT (_S(G95X_S_DEFINED), 1);
        }
      else
        {
	  G95X_ADD_ATT_STR (_S(G95X_S_NAME), sym->name);
        }
 
    }

  dump_symbol_attributes (sym);

  G95X_END_OBJ (symbol, st);
}
#endif

void
G95X_DUMP_FCT (common, g95_symtree * st)
#ifdef INTF
;
#else
{
  g95_symbol *sym;
  g95_common_head *c = st->n.common;


  G95X_ADD_OBJ (_S(G95X_S_COMMON), c);
  G95X_DUMP_ATTR_FCT (common, st);
  if (st->name)
    G95X_ADD_ATT_STR (_S(G95X_S_NAME), st->name);
  else if (g95x_option.xml_defn)
    G95X_ADD_ATT_INT (_S(G95X_S_BLANK), 1);

  if (g95x_option.xml_defn) 
  {
    G95X_ADD_LST (_S(G95X_S_SYMBOL G95X_S_HN G95X_S_LIST));
    for (sym = c->head; sym; sym = sym->common_next)
        G95X_PSH_LST_OBJ (sym);
    G95X_END_LST ();
  }

  G95X_END_OBJ (common, st);

}
#endif

void
G95X_DUMP_FCT (label, g95_st_label * label)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_LABEL), label);
  G95X_DUMP_ATTR_FCT (label, label);
  G95X_ADD_ATT_INT (_S(G95X_S_VALUE), label->value);
  G95X_END_OBJ (label, label);
}
#endif


#ifndef INTF

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
_trs (g95_symtree * st, void *data)
{
  g95_symbol *sym = st->n.sym;
  int glbl = g95x_symbol_global (sym);
//int dfnd = g95x_symbol_defined (sym);

  if (sym->attr.abstract)
    return;
  if (sym->attr.flavor == FL_DERIVED)
    return;
  if (!g95x_symbol_valid (sym))
    return;
  if (g95x_defined_obj_id ((g95x_pointer) sym))
    goto assoc;

  if (glbl)
    goto assoc;

  if (sym->x.imported)
    G95X_DUMP_SUBOBJ_AN (imported_symbol, st);
  else
    G95X_DUMP_SUBOBJ_AN (symbol, st);
assoc:
  if (g95x_symbol_global (sym) && (st->name[0] != '@') &&
     (!g95x_defined_als_id (st->name, (g95x_pointer) sym))
     && strcmp (sym->name,G95_MAIN) )
    G95X_DUMP_SUBOBJ_AN (association, st);
    
}

static void
_tra (g95_symtree * st, void *data)
{
  g95_symbol *sym = st->n.sym;
  int glbl = g95x_symbol_global (sym);
//int dfnd = g95x_symbol_defined (sym);

  if (!sym->attr.abstract)
    return;
  if (!g95x_symbol_valid (sym))
    return;
  if (g95x_defined_obj_id ((g95x_pointer) sym))
    goto assoc;

  if (glbl)
    goto assoc;

  if (sym->x.imported)
    G95X_DUMP_SUBOBJ_AN (imported_abstract, st);
  else
    G95X_DUMP_SUBOBJ_AN (abstract, st);
assoc:
  if (g95x_symbol_global (sym) && (st->name[0] != '@') &&
     (!g95x_defined_als_id (st->name, (g95x_pointer) sym)))
    G95X_DUMP_SUBOBJ_AN (association, st);
    
}

static void
dump_symbols (g95_namespace * ns)
{
  G95X_ADD_LST (_S(G95X_S_SYMBOL G95X_S_HN G95X_S_LIST));
  traverse_symtree (ns->sym_root, _trs, NULL);
  G95X_END_LST ();
}

static void
dump_abstracts (g95_namespace * ns)
{
  G95X_ADD_LST (_S(G95X_S_ABSTRACT G95X_S_HN G95X_S_INTERFACE G95X_S_HN
		G95X_S_LIST));
  traverse_symtree (ns->sym_root, _tra, NULL);
  G95X_END_LST ();
}

static void
_trt (g95_symtree * st, void *data)
{
  g95_symbol *sym = st->n.sym;
  int glbl = g95x_symbol_global (sym);
//int dfnd = g95x_symbol_defined (sym);

  if (sym->attr.flavor != FL_DERIVED)
    return;
  if (g95x_defined_obj_id ((g95x_pointer) sym))
    goto assoc;

  if (glbl)
    goto assoc;

  if (sym->x.imported)
    G95X_DUMP_SUBOBJ_AN (imported_user_type, st);
  else
    G95X_DUMP_SUBOBJ_AN (user_type, st);
assoc:
  if (g95x_symbol_global (sym) && (st->name[0] != '@') &&
     (!g95x_defined_als_id (st->name, (g95x_pointer) sym)))
    G95X_DUMP_SUBOBJ_AN (association, st);
    
}

static void
dump_types (g95_namespace * ns)
{
  G95X_ADD_LST (_S(G95X_S_USER G95X_S_HN G95X_S_TYPE G95X_S_HN G95X_S_LIST));
  traverse_symtree (ns->sym_root, _trt, NULL);
  G95X_END_LST ();
}

static void
_trc (g95_symtree * st, void *data)
{
  if (g95x_common_valid (st->n.common))
    G95X_DUMP_SUBOBJ_AN (common, st);
}

static void
dump_commons (g95_namespace * ns)
{

  G95X_ADD_LST (_S(G95X_S_COMMON G95X_S_HN G95X_S_LIST));

  traverse_symtree (ns->common_root, _trc, NULL);

  if (ns->blank_common)
    {
      g95_symtree st;
      st.name = NULL;
      st.n.common = ns->blank_common;
      G95X_DUMP_SUBOBJ_AN (common, &st);
    }

  G95X_END_LST ();
}

static void
dump_labels (g95_namespace * ns)
{
  g95_st_label *label;

  G95X_ADD_LST (_S(G95X_S_LABEL G95X_S_HN G95X_S_LIST));
  for (label = ns->st_labels; label; label = label->next)
    G95X_DUMP_SUBOBJ_AN (label, label);
  G95X_END_LST ();
}

static void
_trg (g95_symtree * st, void *data)
{
  G95X_DUMP_SUBOBJ_AN (generic, st);
}

static void
dump_generics (g95_namespace * ns)
{
  G95X_ADD_LST (_S(G95X_S_GENERIC G95X_S_HN G95X_S_LIST));
  traverse_symtree (ns->generic_root, _trg, NULL);
  G95X_END_LST ();
}



static void
_tro (g95_symtree * st, void *data)
{
  G95X_DUMP_SUBOBJ_AN (user_operator, st);
}


static void
dump_operators (g95_namespace * ns)
{
  int l;
  g95x_operator op;
  op.ns = ns;
  G95X_ADD_LST (_S(G95X_S_OPERATOR G95X_S_HN G95X_S_LIST));
  for (l = 0; l < G95_INTRINSIC_OPS; l++)
    {
      if ((l == INTRINSIC_USER) || (l == INTRINSIC_ASSIGN))
	continue;
      if (ns->operator[l])
	{
	  op.iop = l;
	  G95X_DUMP_SUBOBJ_AN (intrinsic_operator, &op);
	}
    }
  traverse_symtree (ns->uop_root, _tro, NULL);
  G95X_END_LST ();

  if (ns->operator[INTRINSIC_ASSIGN])
    {
      op.iop = INTRINSIC_ASSIGN;
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ASSIGNMENT), intrinsic_assignment, &op);
    }
}

static void
_eob (g95x_statement ** sta)
{
  g95x_statement *sta1;
  for (sta1 = *sta; sta1;)
    {
      if (sta1->eblock)
	{
	  if (sta1->eblock->fblock == sta1)
	    sta1 = sta1->eblock;
	  else
	    break;
	}
      else
	break;
    }
  *sta = sta1->next;
}



static void
dump_statements (g95x_statement ** sta, g95x_statement * stp)
{
  g95x_statement *sta1;

  sta1 = *sta;

  for (; sta1 != stp;)
    {
      if (sta1->type == ST_CONTAINS)
	{
	  G95X_DUMP_SUBOBJ_AN (contains_block, sta1);
	  G95X_DUMP_SUBOBJ_AN (statement, sta1->next);
	  _eob (&sta1);
	}
      else if (sta1->type == ST_DERIVED_DECL)
	{
	  G95X_DUMP_SUBOBJ_AN (derived_block, sta1);
	  _eob (&sta1);
	}
      else if (sta1->type == ST_INTERFACE)
	{
	  G95X_DUMP_SUBOBJ_AN (interface_block, sta1);
	  _eob (&sta1);
	}
      else if (sta1->type == ST_IF_BLOCK)
	{
	  G95X_DUMP_SUBOBJ_AN (if_block, sta1);
	  _eob (&sta1);
	}
      else if (sta1->type == ST_SELECT_CASE)
	{
	  G95X_DUMP_SUBOBJ_AN (select_block, sta1);
	  _eob (&sta1);
	}
      else if (sta1->type == ST_WHERE_BLOCK)
	{
	  G95X_DUMP_SUBOBJ_AN (where_block, sta1);
	  _eob (&sta1);
	}
      else if (sta1->type == ST_CRITICAL)
	{
	  G95X_DUMP_SUBOBJ_AN (critical_block, sta1);
	  _eob (&sta1);
	}
      else if (sta1->type == ST_FORALL_BLOCK)
	{
	  G95X_DUMP_SUBOBJ_AN (forall_block, sta1);
	  _eob (&sta1);
	}
      else if (sta1->type == ST_DO)
	{
	  G95X_DUMP_SUBOBJ_AN (do_block, sta1);
	  if (sta1->eblock->type == ST_ENDDO)
	    sta1 = sta1->eblock->next;
	  else
	    sta1 = sta1->eblock;
	}
      else if (simple_statement (sta1))
	{
	  G95X_DUMP_SUBOBJ_AN (statement, sta1);
	  while (simple_statement (sta1))
	    {
	      sta1 = sta1->next;
	    }
	  sta1 = sta1->next;
	}
      else
	{
	  G95X_DUMP_SUBOBJ_AN (statement, sta1);
	  sta1 = sta1->next;
	}
    }

  *sta = sta1;
}


#endif




void
G95X_DUMP_FCT (forall_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta1;
  G95X_ADD_OBJ_NI (_S(G95X_S_FORALL G95X_S_HN G95X_S_CONSTRUCT), &sta->block);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_FORALL G95X_S_HN G95X_S_BLOCK), statement_block,
		       sta);
  sta1 = sta->eblock;

  if (sta1->type != ST_END_FORALL)
    g95x_die ("");

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END G95X_S_HN G95X_S_FORALL G95X_S_HN
		       G95X_S_STATEMENT), statement, sta1);

  G95X_END_OBJ (forall_block, sta);
}
#endif


void
G95X_DUMP_FCT (where_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta1;
  G95X_ADD_OBJ_NI (_S(G95X_S_WHERE G95X_S_HN G95X_S_CONSTRUCT), &sta->block);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_WHERE G95X_S_HN G95X_S_BLOCK), statement_block,
		       sta);
  sta1 = sta->eblock;

  if (sta1->type == ST_ELSEWHERE)
    {
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ELSE G95X_S_HN G95X_S_WHERE G95X_S_HN
			   G95X_S_BLOCK), statement_block, sta1);
      sta1 = sta1->eblock;
    }

  if (sta1->type != ST_END_WHERE)
    g95x_die ("");

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END G95X_S_HN G95X_S_WHERE G95X_S_HN
		       G95X_S_STATEMENT), statement, sta1);

  G95X_END_OBJ (where_block, sta);
}
#endif

void
G95X_DUMP_FCT (do_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta1;

  G95X_ADD_OBJ_NI (_S(G95X_S_DO G95X_S_HN G95X_S_CONSTRUCT), &sta->block);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_DO G95X_S_HN G95X_S_BLOCK), statement_block,
		       sta);
  sta1 = sta->eblock;

  if (sta1->type == ST_ENDDO)
    {
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END G95X_S_HN G95X_S_DO G95X_S_HN
			   G95X_S_STATEMENT), statement, sta1);
      sta1 = sta1->next;
    }

  G95X_END_OBJ (do_block, sta);
}
#endif


void
G95X_DUMP_FCT (select_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta1;
  G95X_ADD_OBJ_NI (_S(G95X_S_CASE G95X_S_HN G95X_S_CONSTRUCT), &sta->block);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_SELECT G95X_S_HN G95X_S_CASE G95X_S_HN
		       G95X_S_STATEMENT), statement_block, sta);
  sta1 = sta->next;

  G95X_ADD_LST (_S(G95X_S_CASE G95X_S_HN G95X_S_BLOCK G95X_S_HN G95X_S_LIST));
  for (; sta1->type == ST_CASE; sta1 = sta1->eblock)
    G95X_DUMP_SUBOBJ_AN (statement_block, sta1);
  G95X_END_LST ();

  if (sta1->type != ST_END_SELECT)
    g95x_die ("");

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END G95X_S_HN G95X_S_SELECT G95X_S_HN
		       G95X_S_STATEMENT), statement, sta1);

  G95X_END_OBJ (select_block, sta);

}
#endif

#ifndef INTF
#ifdef G95X_DUMP_BODY
static const char * statement_block_type (const g95x_statement * sta)
{
  const char * p = NULL;
  switch (sta->type)
    {
      case ST_CRITICAL:
          p = _S(G95X_S_CRITICAL G95X_S_HN G95X_S_HEAD);
        break;
      case ST_IF_BLOCK:
          p = _S(G95X_S_IF G95X_S_HN G95X_S_HEAD);
        break;
      case ST_ELSEIF:
          p = _S(G95X_S_ELSE G95X_S_HN G95X_S_IF G95X_S_HN G95X_S_HEAD);
        break;
      case ST_ELSE:
          p = _S(G95X_S_ELSE G95X_S_HN G95X_S_HEAD);
        break;
      case ST_FORALL_BLOCK:
          p = _S(G95X_S_FORALL G95X_S_HN G95X_S_HEAD);
        break;
      case ST_WHERE_BLOCK:
          p = _S(G95X_S_WHERE G95X_S_HN G95X_S_HEAD);
        break;
      case ST_ELSEWHERE:
          p = _S(G95X_S_ELSE G95X_S_HN G95X_S_WHERE G95X_S_HN G95X_S_HEAD);
        break;
      case ST_DO:
          p = _S(G95X_S_DO G95X_S_HN G95X_S_HEAD);
        break;
      case ST_SELECT_CASE:
          p = _S(G95X_S_SELECT G95X_S_HN G95X_S_CASE G95X_S_HN G95X_S_HEAD);
        break;
      case ST_CASE:
          p = _S(G95X_S_CASE G95X_S_HN G95X_S_HEAD);
        break;
      default:
        g95x_die ("");
    }
  return p;
}
#endif
#endif

void
G95X_DUMP_FCT (statement_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta0, *sta1;

  sta0 = sta;
  G95X_ADD_OBJ_NI (_S(G95X_S_BLOCK), &sta->eblock);

  G95X_DUMP_SUBOBJ_NA (statement_block_type (sta), statement, sta); 

  G95X_ADD_LST (_S(G95X_S_STATEMENT G95X_S_HN G95X_S_LIST));

  sta1 = sta->next;

  dump_statements (&sta1, sta0->eblock);
  G95X_END_LST ();
  G95X_END_OBJ (statement_block, sta);
}
#endif



void
G95X_DUMP_FCT (critical_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta1;
  G95X_ADD_OBJ_NI (_S(G95X_S_CRITICAL G95X_S_HN G95X_S_CONSTRUCT), &sta->block);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CRITICAL G95X_S_HN G95X_S_STATEMENT),
		       statement_block, sta);
  sta1 = sta->eblock;

  if (sta1->type != ST_END_CRITICAL)
    g95x_die ("");

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END G95X_S_HN G95X_S_CRITICAL G95X_S_HN
		       G95X_S_STATEMENT), statement, sta1);

  G95X_END_OBJ (critical_block, sta);

}
#endif


void
G95X_DUMP_FCT (if_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta1;
  G95X_ADD_OBJ_NI (_S(G95X_S_IF G95X_S_HN G95X_S_CONSTRUCT), &sta->block);
  G95X_DUMP_ATTR_FCT (if_block, sta);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_IF G95X_S_HN G95X_S_THEN G95X_S_HN G95X_S_BLOCK),
		       statement_block, sta);

  sta1 = sta->eblock;
  G95X_ADD_LST (_S(G95X_S_ELSE G95X_S_HN G95X_S_IF G95X_S_HN G95X_S_BLOCK
		G95X_S_HN G95X_S_LIST));
  for (; sta1->type == ST_ELSEIF; sta1 = sta1->eblock)
    G95X_DUMP_SUBOBJ_AN (statement_block, sta1);
  G95X_END_LST ();

  if (sta1->type == ST_ELSE)
    {
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ELSE G95X_S_HN G95X_S_BLOCK),
			   statement_block, sta1);
      sta1 = sta1->eblock;
    }

  if (sta1->type != ST_ENDIF)
    g95x_die ("");

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END G95X_S_HN G95X_S_IF G95X_S_STATEMENT), statement, sta1);

  G95X_END_OBJ (if_block, sta);

}
#endif


void
G95X_DUMP_FCT (derived_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta1;
  G95X_ADD_OBJ_NI (_S(G95X_S_DERIVED G95X_S_HN G95X_S_BLOCK), &sta->block);
  G95X_DUMP_ATTR_FCT (derived_block, sta);

  G95X_DUMP_SUBOBJ_NA (G95X_S_HEAD, statement, sta);

  G95X_ADD_LST (_S(G95X_S_STATEMENT G95X_S_HN G95X_S_LIST));
  for (sta1 = sta->next; sta1; sta1 = sta1->next)
    {
      if (sta1->type == ST_END_TYPE)
	break;
      G95X_DUMP_SUBOBJ_AN (statement, sta1);
    }
  G95X_END_LST ();

  G95X_DUMP_SUBOBJ_NA (G95X_S_TAIL, statement, sta1);

  G95X_END_OBJ (derived_block, sta);
}
#endif

void
G95X_DUMP_FCT (interface_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *sta1, *stb;
  int intf = 1;

  G95X_ADD_OBJ_NI (_S(G95X_S_INTERFACE G95X_S_HN G95X_S_BLOCK), &sta->block);

  for (sta1 = sta; sta1; sta1 = sta1->next)
    {
      if (sta1->type == ST_INTERFACE)
	{
	  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_INTERFACE G95X_S_HN G95X_S_STATEMENT),
			       statement, sta1);
	  G95X_ADD_LST (_S(G95X_S_INTERFACE G95X_S_HN G95X_S_SPEC G95X_S_HN
			G95X_S_LIST));
	}
      else if (sta1->type == ST_END_INTERFACE)
	{
	  G95X_END_LST ();
	  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END G95X_S_HN G95X_S_INTERFACE G95X_S_HN
			       G95X_S_STATEMENT), statement, sta1);
	}
      else
	{
	  G95X_DUMP_SUBOBJ_AN (statement, sta1);
	}
      intf = (intf || (sta1->type == ST_INTERFACE))
	&& (sta1->type != ST_END_INTERFACE);
      if (!intf)
	break;
      for (stb = sta1->a_next; stb != sta1->next; stb = stb->eblock->a_next)
	if (g95x_valid_namespace (stb->ns))
	  G95X_DUMP_SUBOBJ_AN (namespace, stb->ns);
    }

  G95X_END_OBJ (interface_block, sta);

}
#endif

void
G95X_DUMP_FCT (contains_block, g95x_statement * const sta)
#ifdef INTF
 ;
#else
{
  g95x_statement *stb;
  G95X_ADD_OBJ_NI (_S(G95X_S_INTERNAL G95X_S_HN G95X_S_SUBPROGRAM G95X_S_HN
		   G95X_S_PART), &sta->block);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_CONTAINS G95X_S_HN G95X_S_STATEMENT), statement,
		       sta);
  G95X_ADD_LST (_S(G95X_S_INTERNAL G95X_S_HN G95X_S_SUBPROGRAM G95X_S_HN
		G95X_S_LIST));
  for (stb = sta->a_next; stb != sta->next;)
    {
      if (g95x_valid_namespace (stb->ns))
	G95X_DUMP_SUBOBJ_AN (namespace, stb->ns);
      for (; stb->eblock; stb = stb->eblock);
      stb = stb->a_next;
    }
  G95X_END_LST ();
  G95X_END_OBJ (contains_block, sta);
}
#endif

#ifndef INTF

static void
add_att_access (const char *key, g95_access access)
{
  const char *str = NULL;

  if (access == ACCESS_UNKNOWN)
    return;

  switch (access)
    {
    case ACCESS_PUBLIC:
      str = "PUBLIC";
      break;
    case ACCESS_PRIVATE:
      str = "PRIVATE";
      break;
    default:
      break;
    }
  if (access)
    G95X_ADD_ATT_STR (key, str);
}

#endif




#ifndef INTF

static void
dump_intf_info_block (int n, g95x_intf_info ** iib)
{
  int i, i0, i1;
  g95x_intf_info *iib1[n + 1], *iib0;

again:
  for (i = 0; i < n; i++)
    if (iib[i] != NULL)
      {
	i0 = i;
	goto found;
      }

  return;

found:
  i1 = 0;
  iib0 = iib[i0];
  for (i = i0; i < n; i++)
    if ((iib[i] != NULL) && eq_intf_info (iib[i], iib0))
      {
	iib1[i1++] = iib[i];
	iib[i] = NULL;
      }
    else
      {
	break;
      }
  iib1[i1] = NULL;

  switch (iib1[0]->type)
    {
    case INTERFACE_GENERIC:
      G95X_DUMP_SUBOBJ_AN (intf_info_block_gen, iib1);
      break;
    case INTERFACE_INTRINSIC_OP:
      G95X_DUMP_SUBOBJ_AN (intf_info_block_iop, iib1);
      break;
    case INTERFACE_USER_OP:
      G95X_DUMP_SUBOBJ_AN (intf_info_block_uop, iib1);
      break;
    case INTERFACE_NAMELESS:
    case INTERFACE_ABSTRACT:
      g95x_die ("");
    }

  goto again;

}

static void
dump_procedure_list (g95x_intf_info ** iib)
{
  int i, j;
  G95X_ADD_LST (_S(G95X_S_PROCEDURE G95X_S_HN G95X_S_LIST));
  j = 0;

  for (i = 0; iib[i]; i++)
    if (iib[i]->next == NULL)
      G95X_PSH_LST_OBJ (iib[i]->intf->sym);

#ifdef G95X_DUMP_BODY
  for (j = 0, i = 0; iib[i]; i++)
    if (iib[i]->next)
      iib[j++] = iib[i]->next;
  iib[j] = NULL;
  dump_intf_info_block (j, iib);
#endif

  G95X_END_LST ();
}

#endif



void
G95X_DUMP_FCT (intf_info_block_gen, g95x_intf_info ** iib)
#ifdef INTF
;
#else
{
  char tmp[G95_MAX_SYMBOL_LEN * 2 + 10];
  G95X_ADD_OBJ (_S(G95X_S_GENERIC), iib[0]);
  sprintf (tmp, "%s::%s", iib[0]->module, iib[0]->u.generic_name);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), tmp);
  G95X_DUMP_ATTR_FCT (intf_info_block_gen, iib);
  dump_procedure_list (iib);
  G95X_END_OBJ (intf_info_block_gen, iib);
}
#endif


void
G95X_DUMP_FCT (intf_info_block_iop, g95x_intf_info ** iib)
#ifdef INTF
;
#else
{
  char tmp[G95_MAX_SYMBOL_LEN * 2 + 10];
  G95X_ADD_OBJ (_S(G95X_S_OPERATOR), iib[0]);
  sprintf (tmp, "%s::%s", iib[0]->module, g95x_iop_type (iib[0]->u.iop_idx));
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), tmp);
  G95X_DUMP_ATTR_FCT (intf_info_block_iop, iib);
  dump_procedure_list (iib);
  G95X_END_OBJ (intf_info_block_iop, iib);
}
#endif


void
G95X_DUMP_FCT (intf_info_block_uop, g95x_intf_info ** iib)
#ifdef INTF
;
#else
{
  char tmp[G95_MAX_SYMBOL_LEN * 2 + 10];
  G95X_ADD_OBJ (_S(G95X_S_GENERIC), iib[0]);
  sprintf (tmp, "%s::.%s.", iib[0]->module, iib[0]->u.uop_name);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), tmp);
  G95X_DUMP_ATTR_FCT (intf_info_block_uop, iib);
  dump_procedure_list (iib);
  G95X_END_OBJ (intf_info_block_uop, iib);
}
#endif

#ifndef INTF

static void
collect_intf_info (g95_interface * intf_list)
{
  g95_interface *intf;
  g95x_intf_info **intf_info_block;
  int n = 0;
  int i;

  G95X_ADD_LST (_S(G95X_S_PROCEDURE G95X_S_HN G95X_S_LIST));
  for (intf = intf_list; intf; intf = intf->next)
    if (intf->x.intf_info)
      {
	n++;
      }
    else
      {
	G95X_PSH_LST_OBJ (intf->sym);
      }

  intf_info_block =
    (g95x_intf_info **) g95_getmem (sizeof (g95x_intf_info *) * n);
  for (i = 0, intf = intf_list; intf; intf = intf->next)
    if (intf->x.intf_info)
      intf_info_block[i++] = intf->x.intf_info;

  dump_intf_info_block (n, intf_info_block);

  g95_free (intf_info_block);

  G95X_END_LST ();

}

#endif


void
G95X_DUMP_FCT (generic, g95_symtree * st)
#ifdef INTF
;
#else
{

  G95X_ADD_OBJ (_S(G95X_S_GENERIC), st);
  G95X_DUMP_ATTR_FCT (generic, st);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), st->name);

  add_att_access (_S(G95X_S_ACCESS), st->access);

  if (g95x_option.xml_defn)
    collect_intf_info (st->n.generic);

  G95X_END_OBJ (generic, st);

}
#endif

void
G95X_DUMP_FCT (user_operator, g95_symtree * st)
#ifdef INTF
;
#else
{
  char name[G95_MAX_SYMBOL_LEN + 1];

  G95X_ADD_OBJ (_S(G95X_S_OPERATOR), st->n.uop);
  G95X_DUMP_ATTR_FCT (user_operator, st);

  sprintf (name, ".%s.", st->name);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), name);

  add_att_access (_S(G95X_S_ACCESS), st->access);

  if (g95x_option.xml_defn)
    collect_intf_info (st->n.uop->operator );

  G95X_END_OBJ (user_operator, st);

}
#endif

void
G95X_DUMP_FCT (intrinsic_assignment, g95x_operator * op)
#ifdef INTF
;
#else
{
  g95_namespace *ns = op->ns;
  G95X_ADD_OBJ_T (_S(G95X_S_ASSIGNMENT), _S(G95X_S_INTRINSIC), 
                  &ns->operator[INTRINSIC_ASSIGN]);

  G95X_DUMP_ATTR_FCT (intrinsic_assignment, op);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), g95x_iop_type (INTRINSIC_ASSIGN));

  if (g95x_option.xml_defn)
    collect_intf_info (ns->operator[INTRINSIC_ASSIGN]);

  G95X_END_OBJ (intrinsic_assignment, op);
}
#endif


void
G95X_DUMP_FCT (intrinsic_operator, g95x_operator * op)
#ifdef INTF
;
#else
{
  g95_namespace *ns = op->ns;
  g95_intrinsic_op iop = op->iop;
  G95X_ADD_OBJ_T (_S(G95X_S_OPERATOR), _S(G95X_S_INTRINSIC), &ns->operator[iop]);

  G95X_DUMP_ATTR_FCT (intrinsic_operator, op);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), g95x_iop_type (iop));

  if (g95x_option.xml_defn)
    collect_intf_info (ns->operator[iop]);

  G95X_END_OBJ (intrinsic_operator, op);
}
#endif

void
G95X_DUMP_FCT (namespace, g95_namespace * ns)
#ifdef INTF
;
#else
{
  g95x_statement *sta;
  g95_namespace *save_ns;

  save_ns = g95x_dump_ns;
  g95x_dump_ns = ns;

  if (ns->interface)
    {
      G95X_ADD_OBJ_NI (_S(G95X_S_INTERFACE G95X_S_HN G95X_S_BODY), ns);
    }
  else
    {
      G95X_ADD_OBJ_NI (_S(G95X_S_PROGRAM G95X_S_HN G95X_S_UNIT), ns);
      if (g95x_option.xml_defn)
	G95X_ADD_ATT_STR (_S(G95X_S_TYPE), g95x_namespace_type (ns));
    }

  G95X_DUMP_ATTR_FCT (namespace, ns);

  if (g95x_option.xml_defn)
    G95X_ADD_ATT_INT (_S(G95X_S_IMPORT), ns->import);

  dump_abstracts (ns);

  dump_symbols (ns);

  dump_types (ns);

  dump_commons (ns);

  dump_labels (ns);

  dump_generics (ns);

  dump_operators (ns);

  sta = ns->x.statement_head;
  G95X_ADD_LST (_S(G95X_S_STATEMENT G95X_S_HN G95X_S_LIST));
  dump_statements (&sta, NULL);
  G95X_END_LST ();

  G95X_END_OBJ (namespace, ns);

  g95x_dump_ns = save_ns;

}
#endif

void
G95X_DUMP_FCT (forall_iterator, g95_forall_iterator * fa)
#ifdef INTF
;
#else
{

  G95X_ADD_OBJ_NI (_S(G95X_S_FORALL G95X_S_HN G95X_S_TRIPLET G95X_S_HN
		   G95X_S_SPEC), fa);
  G95X_DUMP_ATTR_FCT (forall_iterator, fa);

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_VARIABLE), expr, fa->var);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_START), expr, fa->start);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_END), expr, fa->end);

  if (g95x_valid_expr (fa->stride))
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_STEP), expr, fa->stride);

  G95X_END_OBJ (forall_iterator, fa);

}
#endif


void
G95X_DUMP_FCT (io_var, g95_code * c)
#ifdef INTF
;
#else
{
  g95_code *d;

  G95X_ADD_OBJ_NI (_S(G95X_S_IO G95X_S_HN G95X_S_VARIABLE), c->block);
  G95X_DUMP_ATTR_FCT (io_var, c);

  G95X_DUMP_WHERE (&c->x.where);

  G95X_ADD_LST (_S(G95X_S_VARIABLE G95X_S_HN G95X_S_LIST));
  for (d = c->block; d; d = d->next)
    {
      switch (d->type)
	{
	case EXEC_DO:
	  G95X_DUMP_SUBOBJ_AN (io_var, d);
	  break;
	case EXEC_TRANSFER:
	  G95X_DUMP_SUBOBJ_AN (expr, d->expr);
	  break;
	default:
	  g95x_die ("");
	  break;
	}
    }
  G95X_END_LST ();

  if (c->ext.iterator)
    G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ITERATOR), iterator, c->ext.iterator);

  G95X_END_OBJ (io_var, c);

}
#endif


void
G95X_DUMP_FCT (locus_section, g95x_locus_section * ls)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_LOCUS G95X_S_HN G95X_S_SECTION), ls);
  G95X_DUMP_ATTR_FCT (locus_section, ls);
  G95X_ADD_ATT_INT (_S(G95X_S_CHAR G95X_S_HN G95X_S_1), ls->xr->column1);
  G95X_ADD_ATT_INT (_S(G95X_S_CHAR G95X_S_HN G95X_S_2), ls->xr->column2);
  G95X_ADD_ATT_INT (_S(G95X_S_LINE), ls->xr->lb1->linenum);
  G95X_ADD_ATT_OBJ (_S(G95X_S_FILE), ls->xr->lb1->file);
  G95X_END_OBJ (locus_section, ls);
}
#endif



void
G95X_DUMP_FCT (locus, g95x_locus * xw)
#ifdef INTF
;
#else
{
  g95x_locus *xr;
  int nxr, i;
  g95x_locus_section ls;

  g95x_refine_location (xw, &xr, &nxr);

  if (nxr == 0)
    {
      G95X_ADD_OBJ (_S(G95X_S_LOCUS G95X_S_HN G95X_S_SIMPLE), xw);
      G95X_DUMP_ATTR_FCT (locus, xw);
      G95X_ADD_ATT_INT (_S(G95X_S_CHAR G95X_S_HN G95X_S_1), xw->column1);
      G95X_ADD_ATT_INT (_S(G95X_S_CHAR G95X_S_HN G95X_S_2), xw->column2);
      G95X_ADD_ATT_INT (_S(G95X_S_LINE), xw->lb1->linenum);
      G95X_ADD_ATT_OBJ (_S(G95X_S_FILE), xw->lb1->file);
      G95X_END_OBJ (locus, xw);
    }
  else if (nxr == 1)
    {
      G95X_ADD_OBJ (_S(G95X_S_LOCUS G95X_S_HN G95X_S_SIMPLE), xw);
      G95X_DUMP_ATTR_FCT (locus, xw);
      G95X_ADD_ATT_INT (_S(G95X_S_CHAR G95X_S_HN G95X_S_1), xr->column1);
      G95X_ADD_ATT_INT (_S(G95X_S_CHAR G95X_S_HN G95X_S_2), xr->column2);
      G95X_ADD_ATT_INT (_S(G95X_S_LINE), xr->lb1->linenum);
      G95X_ADD_ATT_OBJ (_S(G95X_S_FILE), xr->lb1->file);
      G95X_END_OBJ (locus, xw);
    }
  else
    {
      G95X_ADD_OBJ (_S(G95X_S_LOCUS), xw);
      G95X_ADD_LST (_S(G95X_S_SECTION G95X_S_HN G95X_S_LIST));
      for (i = 0; i < nxr; i++)
	{
	  ls.xr = &xr[i];
	  G95X_DUMP_SUBOBJ_AN (locus_section, &ls);
	}
      G95X_END_LST ();
      G95X_END_OBJ (locus, xw);
    }
}
#endif



#ifndef INTF

static void
dump_intrinsic_types ()
{
  g95_typespec ts;
  memset (&ts, 0, sizeof (ts));

  G95X_ADD_LST (_S(G95X_S_INTRINSIC G95X_S_HN G95X_S_TYPE G95X_S_HN G95X_S_LIST));
  ts.type = BT_INTEGER;
  G95X_DUMP_SUBOBJ_AN (intrinsic_type, g95x_type_id (&ts));
  ts.type = BT_REAL;
  G95X_DUMP_SUBOBJ_AN (intrinsic_type, g95x_type_id (&ts));
  ts.type = BT_COMPLEX;
  G95X_DUMP_SUBOBJ_AN (intrinsic_type, g95x_type_id (&ts));
  ts.type = BT_LOGICAL;
  G95X_DUMP_SUBOBJ_AN (intrinsic_type, g95x_type_id (&ts));
  ts.type = BT_CHARACTER;
  G95X_DUMP_SUBOBJ_AN (intrinsic_type, g95x_type_id (&ts));
  ts.type = BT_PROCEDURE;
  G95X_DUMP_SUBOBJ_AN (intrinsic_type, g95x_type_id (&ts));
  ts.x.dble = 1;
  ts.type = BT_REAL;
  G95X_DUMP_SUBOBJ_AN (intrinsic_type, g95x_type_id (&ts));
  ts.type = BT_COMPLEX;
  G95X_DUMP_SUBOBJ_AN (intrinsic_type, g95x_type_id (&ts));
  G95X_END_LST ();

}

static void
dump_intrinsic_operators ()
{

  G95X_ADD_LST (_S(G95X_S_INTRINSIC G95X_S_HN G95X_S_OPERATOR G95X_S_HN
		G95X_S_LIST));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_PLUS));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_MINUS));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_TIMES));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_DIVIDE));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_POWER));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_CONCAT));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_AND));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base, g95x_iop_type (INTRINSIC_OR));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_EQV));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_NEQV));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base, g95x_iop_type (INTRINSIC_EQ));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base, g95x_iop_type (INTRINSIC_NE));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base, g95x_iop_type (INTRINSIC_GT));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base, g95x_iop_type (INTRINSIC_GE));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base, g95x_iop_type (INTRINSIC_LT));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base, g95x_iop_type (INTRINSIC_LE));
  G95X_DUMP_SUBOBJ_AN (intrinsic_operator_base,
		       g95x_iop_type (INTRINSIC_NOT));
  G95X_END_LST ();

  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_INTRINSIC G95X_S_HN G95X_S_ASSIGNMENT),
		       intrinsic_assignment_base,
		       g95x_iop_type (INTRINSIC_ASSIGN));

}

static void
dump_intrinsics ()
{
  g95_intrinsic_sym *f, *s;
  int nf, ns, i;
  g95x_get_intrinsics_info (&f, &s, &nf, &ns);

  G95X_ADD_LST (_S(G95X_S_INTRINSIC G95X_S_HN G95X_S_GENERIC G95X_S_HN
		G95X_S_LIST));
  for (i = 0; i < nf; i++)
    if (f[i].x.used && strcmp (f[i].name, ""))
      G95X_DUMP_SUBOBJ_AN (intrinsic_gen, &f[i]);
  for (i = 0; i < ns; i++)
    if (s[i].x.used && strcmp (s[i].name, ""))
      G95X_DUMP_SUBOBJ_AN (intrinsic_gen, &s[i]);
  G95X_END_LST ();

  G95X_ADD_LST (_S(G95X_S_INTRINSIC G95X_S_HN G95X_S_SYMBOL G95X_S_HN
		G95X_S_LIST));
  for (i = 0; i < nf; i++)
    if (f[i].x.used)
      G95X_DUMP_SUBOBJ_AN (intrinsic_sym, &f[i]);
  for (i = 0; i < ns; i++)
    if (s[i].x.used)
      G95X_DUMP_SUBOBJ_AN (intrinsic_sym, &s[i]);
  G95X_END_LST ();


}


#endif




void
G95X_DUMP_FCT (intrinsic_gen, g95_intrinsic_sym * is)
#ifdef INTF
;
#else
{

  G95X_ADD_OBJ (_S(G95X_S_GENERIC), &is->generic);
  G95X_DUMP_ATTR_FCT (intrinsic_gen, is);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), is->name);
  G95X_ADD_LST (_S(G95X_S_SPECIFIC G95X_S_HN G95X_S_LIST));
  G95X_PSH_LST_OBJ (is);
  if (is->generic)
    for (is = is->specific_head; is; is = is->next)
      G95X_PSH_LST_OBJ (is);
  G95X_END_LST ();
  G95X_END_OBJ (intrinsic_gen, is);
}
#endif


void
G95X_DUMP_FCT (intrinsic_sym, g95_intrinsic_sym * is)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_INTRINSIC), is);
  G95X_DUMP_ATTR_FCT (intrinsic_sym, is);
  if (strcmp (is->name, ""))
    G95X_ADD_ATT_STR (_S(G95X_S_NAME), is->name);
  G95X_END_OBJ (intrinsic_sym, is);
}
#endif

void
G95X_DUMP_FCT (intrinsic_assignment_base, const void *typeid)
#ifdef INTF
 ;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_ASSIGNMENT), typeid);
  G95X_DUMP_ATTR_FCT (intrinsic_assignment_base, typeid);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), typeid);
  G95X_ADD_ATT_INT (_S(G95X_S_INTRINSIC), 1);
  G95X_END_OBJ (intrinsic_assignment_base, typeid);
}
#endif



void
G95X_DUMP_FCT (intrinsic_operator_base, const void *typeid)
#ifdef INTF
 ;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_OPERATOR), typeid);
  G95X_DUMP_ATTR_FCT (intrinsic_operator_base, typeid);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), typeid);
  G95X_ADD_ATT_INT (_S(G95X_S_INTRINSIC), 1);
  G95X_END_OBJ (intrinsic_operator_base, typeid);
}
#endif


void
G95X_DUMP_FCT (intrinsic_type, const void *typeid)
#ifdef INTF
 ;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_TYPE), typeid);
  G95X_DUMP_ATTR_FCT (intrinsic_type, typeid);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), typeid);
  G95X_ADD_ATT_INT (_S(G95X_S_INTRINSIC), 1);
  G95X_END_OBJ (intrinsic_type, typeid);
}
#endif


void
G95X_DUMP_FCT (intrinsic_type_name, g95x_intrinsic_type_name * itn)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_NI (_S(G95X_S_INTRINSIC G95X_S_HN G95X_S_TYPE G95X_S_HN G95X_S_NAME),
                   itn);

  G95X_DUMP_ATTR_FCT (intrinsic_type_name, itn);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), itn->name);

  G95X_END_OBJ (intrinsic_type_name, itn);
}
#endif

void
G95X_DUMP_FCT (typespec, g95_typespec * ts)
#ifdef INTF
;
#else
{
  g95_expr * kind, * len;
  g95_charlen * cl;
  const void * typeid;
  g95x_item_locus * lst;
  g95x_temp * temp;
  const char * typenm;
  const char * typeN;
  

  lst = NULL;
  temp = NULL;

  kind = len = NULL;

  get_tklc (ts, &typeid, &kind, &len, &cl);

  typenm = g95x_type_name (ts);

  if (ts->derived) 
    {
      typeN = _S(G95X_S_DERIVED);
    } 
  else if (ts->interface) 
    {
      typeN = _S(G95X_S_INTERFACE);
    }
  else 
    {
      typeN = _S(G95X_S_INTRINSIC);
    }

  G95X_ADD_OBJ_T_NI (_S(G95X_S_TYPE G95X_S_HN G95X_S_SPEC), typeN, ts);


  G95X_DUMP_ATTR_FCT (typespec, ts);
  
  if (g95x_option.xml_defn)
    G95X_ADD_ATT_OBJ (_S(G95X_S_TYPE), typeid);

  G95X_DUMP_WHERE (&ts->x.where);

/*=
      ABSTRACT INTERFACE
        SUBROUTINE INTF
        END SUBROUTINE
      END INTERFACE
      TYPE POINT
        INTEGER :: R, G, B
      END TYPE
      REAL    :: X1                 !
      REAL*4  :: X2                 !
      REAL(4) :: X3                 !
      TYPE(POINT) :: P              !
      PROCEDURE(INTF), POINTER :: F !
      CHARACTER*8             :: S1 !         
      CHARACTER(LEN=8)        :: S2 !
      CHARACTER(LEN=8,KIND=1) :: S3 !
      CHARACTER(8,1)          :: S4 !
      CHARACTER(KIND=1,LEN=8) :: S5 !
      CHARACTER(8)            :: S6 !
=*/

  if (ts->derived) 
    {
      dump_local_symbol_ref_na (_S(G95X_S_USER G95X_S_HN G95X_S_TYPE G95X_S_HN
  			        G95X_S_NAME), ts->derived, &ts->x.where_derived);
    } 
  else if (ts->interface) 
    {
      dump_local_symbol_ref_na (_S(G95X_S_INTERFACE G95X_S_HN G95X_S_NAME),
			        ts->interface, &ts->x.where_interface);
    }
  else 
    {
      char name[G95_MAX_SYMBOL_LEN+1];
      g95x_intrinsic_type_name itn;

      itn.where = ts->x.where;
      itn.name = typenm;
      g95x_get_name_substr (&itn.where, name);

      if (strcasecmp(name, typenm))
        g95x_die ("");
      G95X_DUMP_SUBOBJ_NA (_S(G95X_S_INTRINSIC G95X_S_HN G95X_S_TYPE G95X_S_HN G95X_S_NAME), 
                           intrinsic_type_name, &itn);
    }

  if (kind)
    G95X_PSH_LOC_NA (ts->x.star1 ? _S(G95X_S_SIZE G95X_S_HN G95X_S_SELECTOR) : 
                     _S(G95X_S_KIND G95X_S_HN G95X_S_SELECTOR), 
                     expr, kind, lst);

  if (len)
    G95X_PSH_LOC_NA (_S(G95X_S_CHAR G95X_S_HN G95X_S_SELECTOR), expr, len, lst);
  else if (cl && (cl->length == NULL))
    G95X_PSH_LOC_NA (_S(G95X_S_CHAR G95X_S_HN G95X_S_SELECTOR), expr_chst,
		     get_temp_expr_chst (&temp, cl, &cl->x.chst_where), lst);

  dump_lst_na (lst);

  free_temp (temp);

  G95X_END_OBJ (typespec, ts);
}
#endif

void
G95X_DUMP_FCT (executable_program, g95x_executable_program * prog)
#ifdef INTF
;
#else
{
  g95_namespace *ns;


  G95X_ADD_OBJ_NI (_S(G95X_S_EXECUTABLE G95X_S_HN G95X_S_PROGRAM), prog);
  G95X_DUMP_ATTR_FCT (executable_program, prog);

  if( g95x_option.xml_defn ) {
    dump_intrinsic_types ();
    dump_intrinsic_operators ();
    dump_intrinsics ();
  }

  G95X_ADD_LST (_S(G95X_S_PROGRAM G95X_S_HN G95X_S_UNIT G95X_S_HN G95X_S_LIST));
  for (ns = prog->head; ns; ns = ns->sibling)
    if (g95x_valid_namespace (ns))
      G95X_DUMP_SUBOBJ_AN (namespace, ns);
  G95X_END_LST ();

  G95X_END_OBJ (executable_program, prog);
}
#endif


void
G95X_DUMP_FCT (user_operator1_ref, g95x_user_operator1_ref * uop)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_OPERATOR G95X_S_HN G95X_S_SPEC), uop);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_OPERATOR), symbol_ref, &uop->sr);
  G95X_END_OBJ (user_operator1_ref, uop);
}
#endif


void
G95X_DUMP_FCT (intrinsic_operator1_ref, g95x_intrinsic_operator1_ref * iop)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_OPERATOR G95X_S_HN G95X_S_SPEC), iop);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_OPERATOR), symbol_ref, &iop->sr);
  G95X_END_OBJ (intrinsic_operator1_ref, iop);
}
#endif

void
G95X_DUMP_FCT (assignment1_ref, g95x_assignment1_ref * as)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_ASSIGNMENT G95X_S_HN G95X_S_SPEC), as);
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_ASSIGNMENT), symbol_ref, &as->sr);
  G95X_END_OBJ (assignment1_ref, as);
}
#endif


void
G95X_DUMP_FCT (position_spec_unit, g95_expr* g)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ (_S(G95X_S_POSITION G95X_S_HN G95X_S_SPEC), g);
  G95X_ADD_ATT_OBJ (_S(G95X_S_NAME), _S(G95X_S_UNIT));
  G95X_DUMP_SUBOBJ_NA (_S(G95X_S_VALUE), expr, g);
  G95X_END_OBJ (position_spec_unit, g);
}
#endif


void
G95X_DUMP_FCT (global_symbol, g95x_global_symbol* g)
#ifdef INTF
;
#else
{
  if (g->sym->attr.abstract)
    G95X_ADD_OBJ_T_NI (G95X_S_ABSTRACT, _S(G95X_S_GLOBAL), g);
  else
    G95X_ADD_OBJ_T_NI (g95x_flavor_type (g->sym), _S(G95X_S_GLOBAL), g);

//G95X_ADD_OBJ_T_NI (_S(G95X_S_GLOBAL), g95x_flavor_type (g->sym), g);
//G95X_ADD_OBJ_NI (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_SYMBOL), g);

  G95X_ADD_ATT_STR (_S(G95X_S_FLAVOR), g95x_flavor_type(g->sym));
  G95X_DUMP_ATTR_FCT (global_symbol, g);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), g->name);

  G95X_ADD_ATT_INT (_S(G95X_S_ALLOCATABLE), g->sym->attr.allocatable);
  G95X_ADD_ATT_INT (_S(G95X_S_DERIVED), g->sym->ts.derived ? 1 : 0);
  
  if (g->sym->ts.derived)
    G95X_ADD_ATT_STR (_S(G95X_S_TYPE), g->sym->ts.derived->name);
  else
  {
    const void* type = g95x_type_id(&g->sym->ts.type);
    if (type) G95X_ADD_ATT_STR (_S(G95X_S_TYPE), type);
  }
  
  G95X_ADD_ATT_INT (_S(G95X_S_KIND), g->sym->ts.kind);

  if (g->defined)
    G95X_ADD_ATT_INT (_S(G95X_S_DEFINED), g->defined);

  if (g->sym->as)
  {
    G95X_ADD_ATT_INT (_S(G95X_S_DIMENSION), g->sym->as->rank);
    
    g95x_dump_source_enabled = 0;
    
    if (!g->sym->attr.allocatable && g->sym->as->rank && (g->sym->as->type == AS_EXPLICIT))
    {     
      int i;
      g95x_shape_spec ss;

      G95X_ADD_SCT(_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC)); 
      G95X_ADD_OBJ_NI (_S(G95X_S_ARRAY G95X_S_HN G95X_S_SPEC), g->sym->as);

      G95X_ADD_ATT_STR (_S(G95X_S_SHAPE), g95x_array_spec_type (g->sym->as));
      G95X_DUMP_ATTR_FCT (array_spec, g->sym->as);

      G95X_DUMP_WHERE (&as->x.where);

      G95X_ADD_LST (_S(G95X_S_SHAPE G95X_S_HN G95X_S_SPEC G95X_S_HN G95X_S_LIST));
      for (i = 0; i < g->sym->as->rank; i++)
      {
        ss.as = g->sym->as;
        ss.i = i;
        g95_array_spec *as = ss.as;
        g95_expr *lower = as->lower[i];
        g95_expr *upper = as->upper[i];

        G95X_FULLIFY (lower);
        G95X_FULLIFY (upper);

        G95X_ADD_OBJ_NI (_S(G95X_S_SHAPE G95X_S_HN G95X_S_SPEC), lower);
        G95X_DUMP_ATTR_FCT (shape_spec, &ss);
        G95X_DUMP_WHERE (&as->x.shape_spec_where[i]);

        G95X_DUMP_SUBOBJ_NA (_S(G95X_S_LOWER G95X_S_HN G95X_S_BOUND), expr, lower);
        G95X_DUMP_SUBOBJ_NA (_S(G95X_S_UPPER G95X_S_HN G95X_S_BOUND), expr, upper);

        G95X_END_OBJ (shape_spec, &ss);
      }
      G95X_END_LST ();

      G95X_END_OBJ (array_spec, g->sym->as);
      G95X_END_SCT();
    }

    g95x_dump_source_enabled = 1;
  }

  G95X_END_OBJ (global_symbol, g);
}
#endif


void
G95X_DUMP_FCT (global_common, g95x_global_common* g)
#ifdef INTF
;
#else
{
  G95X_ADD_OBJ_NI (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_COMMON), g);
  G95X_DUMP_ATTR_FCT (global_common, g);
  G95X_ADD_ATT_STR (_S(G95X_S_NAME), g->name);
  G95X_END_OBJ (global_symbol, g);
}
#endif

#ifndef INTF
#ifdef G95X_DUMP_BODY

static g95x_bbt_id_root global_root_bbt_id = { NULL, 0 };

typedef struct {
  int dfnd;
  enum {
    _dump_global_trs,
    _dump_global_trt,
    _dump_global_trc,
    _dump_global_tra
  } type;
} _dump_global_opts;

static void dump_global_c_1 (g95_symtree *st, void* data)
{
  g95x_global_common gcom;
  g95_common_head* c = st->n.common;
  if (!g95x_common_valid (c))
    return;
  if (g95x_get_bbt_id(0, st->name, &global_root_bbt_id))
    return;
  g95x_new_bbt_id(0, st->name, &global_root_bbt_id);
  gcom.name    = st->name;
  gcom.com     = c;
  G95X_DUMP_SUBOBJ_AN (global_common, &gcom);
}

static void dump_global_s_1 (g95_symtree *st, void* data)
{
  g95x_global_symbol gsym;
  _dump_global_opts* dgo = data;
  g95_symbol* sym = st->n.sym;
  int glbl, dfnd;
  const char* gname;
  glbl = g95x_symbol_global (sym);
  if (!glbl)
    return;
  dfnd = g95x_symbol_defined (sym);
  if (dgo->dfnd != dfnd)
    return;
  gname = g95x_symbol_gname(sym);
  if (g95x_get_bbt_id(0, gname, &global_root_bbt_id))
    return;
  g95x_new_bbt_id(0, gname, &global_root_bbt_id);
  gsym.name    = gname;
  gsym.defined = dfnd;
  gsym.sym     = sym;
  G95X_DUMP_SUBOBJ_AN (global_symbol, &gsym);
}

static void dump_global_s_0 (g95_namespace* ns, void* data) 
{
  traverse_symtree (ns->sym_root, dump_global_s_1, data); 
}

static void dump_global_c_0 (g95_namespace* ns, void* data) 
{
  traverse_symtree (ns->common_root, dump_global_c_1, data); 
}

static void traverse_ns (g95_namespace *ns, void (*cb)(g95_namespace*, void*), void* data )
{
  for (; ns; ns = ns->sibling)
    if (g95x_valid_namespace (ns)) 
    {
      cb (ns, data);
      traverse_ns (ns->contained, cb, data);
    }
}

void g95x_dump (g95_namespace *head) 
{
  _dump_global_opts dgo;
  g95x_executable_program prog;
  g95_file* ftop = g95x_get_file_top ();
  prog.head = head;
  prog.where.lb1 = g95x_get_line_head (); prog.where.column1 = 1;
  prog.where.lb2 = g95x_get_line_tail (); prog.where.column2 = prog.where.lb2->size+1;
  g95x_start ();

  G95X_ADD_LST (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_SYMBOL G95X_S_HN G95X_S_LIST));
  dgo.dfnd = 1;
  traverse_ns (head, dump_global_s_0, &dgo);
  dgo.dfnd = 0;
  traverse_ns (head, dump_global_s_0, &dgo);
  G95X_END_LST ();

  G95X_ADD_LST (_S(G95X_S_GLOBAL G95X_S_HN G95X_S_COMMON G95X_S_HN G95X_S_LIST));
  dgo.dfnd = 1;
  traverse_ns (head, dump_global_c_0, &dgo);
  dgo.dfnd = 0;
  traverse_ns (head, dump_global_c_0, &dgo);
  G95X_END_LST ();

  g95x_free_bbt_ids (&global_root_bbt_id);
  g95x_add_fle (ftop);
  g95x_dump_executable_program_body (&prog);
  g95x_close_file (ftop);
  g95x_end();
}
#endif
#endif
