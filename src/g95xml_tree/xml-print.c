#include "g95.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "xml-string.h"



#ifdef G95X_64BITS
const char *REFFMT0 = "0x%llx";
const char *REFFMT1 = "%16.16llx";
#else
const char *REFFMT0 = "0x%x";
const char *REFFMT1 = "%8.8x";
#endif



const char *
g95x_form_type (g95_source_form form)
{
  switch (form)
    {
    case FORM_FIXED:
      return G95X_S_FIXED;
      break;
    case FORM_FREE:
      return G95X_S_FREE;
      break;
    case FORM_UNKNOWN:
      break;
    }
  g95x_die ("");
  return NULL;
}

const char *
g95x_namespace_type (g95_namespace * ns)
{
  g95_symbol *sym = ns->proc_name;

  if (ns->interface)
    {
      if (sym->ts.type == BT_PROCEDURE)
	return G95X_S_SUBROUTINE G95X_S_HN G95X_S_INTERFACE G95X_S_HN
	  G95X_S_BODY;
      else
	return G95X_S_FUNCTION G95X_S_HN G95X_S_INTERFACE G95X_S_HN
	  G95X_S_BODY;
    }

  switch (sym->attr.flavor)
    {
    case FL_PROGRAM:
      return G95X_S_MAIN G95X_S_HN G95X_S_PROGRAM;
    case FL_MODULE:
      return G95X_S_MODULE;
    case FL_PROCEDURE:
      if (sym->ts.type == BT_PROCEDURE)
	{
	  if (sym->attr.external || (ns->parent == NULL))
	    return G95X_S_EXTERNAL G95X_S_HN G95X_S_SUBROUTINE G95X_S_HN
	      G95X_S_SUBPROGRAM;
	  else
	    return G95X_S_INTERNAL G95X_S_HN G95X_S_SUBROUTINE G95X_S_HN
	      G95X_S_SUBPROGRAM;
	}
      else
	{
	  if (g95x_symbol_global (sym))
	    return G95X_S_EXTERNAL G95X_S_HN G95X_S_FUNCTION G95X_S_HN
	      G95X_S_SUBPROGRAM;
	  else
	    return G95X_S_INTERNAL G95X_S_HN G95X_S_FUNCTION G95X_S_HN
	      G95X_S_SUBPROGRAM;
	}
    case FL_BLOCK_DATA:
      return G95X_S_BLOCK G95X_S_HN G95X_S_DATA;
    default:
      g95x_die ("");
    }
  return NULL;
}

int
g95x_symbol_global (g95_symbol * sym)
{


  if (sym->attr.dummy)
    return 0;
  if (sym->x.imported)
    return 1;

  if (sym->module)
    {
      if (!strcmp (sym->module, "(intrinsic)"))
	{
	  return 1;
	}
      else if (!strcmp (sym->module, "(global)"))
	{
	  return 1;
	}
      else if (sym->ns && !strcmp (sym->module, sym->ns->proc_name->name))
	{
	  return 1;
	}
    }

  if (sym->attr.abstract)
    return 0;

  if (sym->attr.access == ACCESS_PRIVATE)
    {
      if ((sym->attr.flavor == FL_PROCEDURE)
	  && (sym->attr.proc == PROC_MODULE))
	{
	  return 1;
	}
      return 0;
    }

  switch (sym->attr.flavor)
    {
    case FL_PROCEDURE:
      switch (sym->attr.proc)
	{
	case PROC_INTERNAL:
	case PROC_DUMMY:
	case PROC_ST_FUNCTION:
	  return 0;
	case PROC_MODULE:
	case PROC_INTRINSIC:
	case PROC_EXTERNAL:
	  return 1;
	default:
	  return 1;
	}
    case FL_PROGRAM:
    case FL_MODULE:
    case FL_BLOCK_DATA:
      return 1;
    default:
      return 0;
    }

}

int
g95x_symbol_defined (g95_symbol * sym)
{
  if (g95x_symbol_intrinsic (sym))
    return 0;
  if (sym->x.imported)
    return 0;
  if (sym->attr.abstract)
    return 1;
  if (sym->formal_ns)
    return 0;
  if (sym->attr.proc == PROC_EXTERNAL)
    return 0;
  if (sym->attr.external)
    return 0;
  return 1;
}

const char *
g95x_symbol_gname (g95_symbol * sym)
{
  static char name[3 * G95_MAX_SYMBOL_LEN + 10];


  if (sym->attr.abstract)
    {
      if (sym->x.imported)
	{
	  sprintf (name, "%s::%s", sym->x.module_name, sym->name);
	}
      else if (sym->ns)
	{
	  g95_namespace *ns = sym->ns;
	  if (ns->proc_name)
	    {
	      g95_symbol *pn = ns->proc_name;
	      if (pn->attr.flavor == FL_MODULE)
		{
		  sprintf (name, "%s::%s", pn->name, sym->name);
		}
	      else
		{
		  g95x_die ("");
		}
	    }
	  else
	    {
	      g95x_die ("");
	    }
	}
      else
	{
	  g95x_die ("");
	}
      return name;
    }

  if (sym->x.imported)
    {
      if (sym->module && (sym->module[0] != '('))
	sprintf (name, "%s::%s", sym->module, sym->name);
      else
	sprintf (name, "%s", sym->name);
      return name;
    }

  if (sym->formal_ns || (sym->attr.proc == PROC_EXTERNAL))
    {
      strcpy (name, sym->name);
      return name;
    }
  if (sym->module && (sym->module[0] != '('))
    {
      sprintf (name, "%s::%s", sym->module, sym->name);
      return name;
    }

  sprintf (name, sym->name);
  return name;
}


int
g95x_symbol_valid (g95_symbol * sym)
{

  if (sym->attr.flavor == FL_UNKNOWN)
    return 0;

  if (sym->x.is_generic)
    return 0;

  if (g95x_symbol_intrinsic (sym))
    return 0;


  return 1;
}

const char *
g95x_statement_type (g95x_statement * sta)
{
  const char *p = NULL;

  switch (sta->type)
    {
    case ST_ARITHMETIC_IF:
      p = G95X_S_ARITHMETIC G95X_S_HN G95X_S_IF;
      break;
    case ST_ALLOCATE:
      p = G95X_S_ALLOCATE;
      break;
    case ST_ATTR_DECL:
      {
	symbol_attribute *a = &sta->u.data_decl.attr;
	if (a->access == ACCESS_PUBLIC)
	  p = G95X_S_PUBLIC;
	if (a->access == ACCESS_PRIVATE)
	  p = G95X_S_PRIVATE;
	if (a->allocatable)
	  p = G95X_S_ALLOCATABLE;
	if (a->dimension)
	  p = G95X_S_DIMENSION;
	if (a->external)
	  p = G95X_S_EXTERNAL;
	if (a->intent == INTENT_IN)
	  p = G95X_S_INTENT_IN_;
	if (a->intent == INTENT_OUT)
	  p = G95X_S_INTENT_OUT_;
	if (a->intent == INTENT_INOUT)
	  p = G95X_S_INTENT_INOUT_;
	if (a->intrinsic)
	  p = G95X_S_INTRINSIC;
	if (a->optional)
	  p = G95X_S_OPTIONAL;
	if (a->pointer)
	  p = G95X_S_POINTER;
	if (a->save)
	  p = G95X_S_SAVE;
	if (a->target)
	  p = G95X_S_TARGET;
	if (a->protected)
	  p = G95X_S_PROTECTED;
	if (a->bind)
	  p = G95X_S_BIND;
	if (a->value)
	  p = G95X_S_VALUE;
	if (a->volatile_)
	  p = G95X_S_VOLATILE;
	if (a->async)
	  p = G95X_S_ASYNCHRONOUS;

	if (p == NULL)
	  goto err;


	break;
      err:
	g95x_die ("Could not find out statement type\n");
      }
    case ST_BACKSPACE:
      p = G95X_S_BACKSPACE;
      break;
    case ST_BLOCK_DATA:
      p = G95X_S_BLOCK G95X_S_HN G95X_S_DATA;
      break;
    case ST_CALL:
      p = G95X_S_CALL;
      break;
    case ST_CASE:
      p = G95X_S_CASE;
      break;
    case ST_CLOSE:
      p = G95X_S_CLOSE;
      break;
    case ST_COMMON:
      p = G95X_S_COMMON;
      break;
    case ST_CONTINUE:
      p = G95X_S_CONTINUE;
      break;
    case ST_CONTAINS:
      p = G95X_S_CONTAINS;
      break;
    case ST_CYCLE:
      p = G95X_S_CYCLE;
      break;
    case ST_DATA_DECL:
      {
	if (sta->in_derived)
	  p = G95X_S_COMPONENT G95X_S_HN G95X_S_DEF;
	else
	  p = G95X_S_TYPE G95X_S_HN G95X_S_DECL;
	break;
      }
    case ST_DATA:
      p = G95X_S_DATA;
      break;
    case ST_DEALLOCATE:
      p = G95X_S_DEALLOCATE;
      break;
    case ST_DERIVED_DECL:
      p = G95X_S_DERIVED G95X_S_HN G95X_S_TYPE G95X_S_HN G95X_S_DEF;
      break;
    case ST_DO:
      p = G95X_S_DO;
      break;
    case ST_ELSE:
      p = G95X_S_ELSE;
      break;
    case ST_ELSEIF:
      p = G95X_S_ELSE G95X_S_HN G95X_S_IF;
      break;
    case ST_ELSEWHERE:
      p = G95X_S_ELSE G95X_S_HN G95X_S_WHERE;
      break;
    case ST_END_BLOCK_DATA:
      p = G95X_S_END G95X_S_HN G95X_S_BLOCK G95X_S_HN G95X_S_DATA;
      break;
    case ST_ENDDO:
      p = G95X_S_END G95X_S_HN G95X_S_DO;
      break;
    case ST_END_ENUM:
      p = G95X_S_END G95X_S_HN G95X_S_ENUM;
      break;
    case ST_END_FILE:
      p = G95X_S_ENDFILE;
      break;
    case ST_END_FORALL:
      p = G95X_S_END G95X_S_HN G95X_S_FORALL;
      break;
    case ST_END_FUNCTION:
      p = G95X_S_END G95X_S_HN G95X_S_FUNCTION;
      break;
    case ST_ENDIF:
      p = G95X_S_END G95X_S_HN G95X_S_IF;
      break;
    case ST_END_INTERFACE:
      p = G95X_S_END G95X_S_HN G95X_S_INTERFACE;
      break;
    case ST_END_MODULE:
      p = G95X_S_END G95X_S_HN G95X_S_MODULE;
      break;
    case ST_END_PROGRAM:
      p = G95X_S_END G95X_S_HN G95X_S_PROGRAM;
      break;
    case ST_END_SELECT:
      p = G95X_S_END G95X_S_HN G95X_S_SELECT G95X_S_HN G95X_S_CASE;
      break;
    case ST_END_SUBROUTINE:
      p = G95X_S_END G95X_S_HN G95X_S_SUBROUTINE;
      break;
    case ST_END_WHERE:
      p = G95X_S_END G95X_S_HN G95X_S_WHERE;
      break;
    case ST_END_TYPE:
      p =
	G95X_S_END G95X_S_HN G95X_S_DERIVED G95X_S_HN G95X_S_TYPE G95X_S_HN
	G95X_S_DEF;
      break;
    case ST_ENTRY:
      p = G95X_S_ENTRY;
      break;
    case ST_ENUM:
      p = G95X_S_ENUM;
      break;
    case ST_ENUMERATOR:
      p = G95X_S_ENUMERATOR;
      break;
    case ST_EQUIVALENCE:
      p = G95X_S_EQUIVALENCE;
      break;
    case ST_EXIT:
      p = G95X_S_EXIT;
      break;
    case ST_FLUSH:
      p = G95X_S_FLUSH;
      break;
    case ST_FORALL_BLOCK:
      p = G95X_S_FORALL G95X_S_HN G95X_S_CONSTRUCT;
      break;
    case ST_FORALL:
      p = G95X_S_FORALL;
      break;
    case ST_FORMAT:
      p = G95X_S_FORMAT;
      break;
    case ST_FUNCTION:
      p = G95X_S_FUNCTION;
      break;
    case ST_GOTO:
      p = G95X_S_GOTO;
      break;
    case ST_IF_BLOCK:
      p = G95X_S_IF G95X_S_HN G95X_S_THEN;
      break;
    case ST_IMPLICIT:
      p = G95X_S_IMPLICIT;
      break;
    case ST_IMPLICIT_NONE:
      p = G95X_S_IMPLICIT G95X_S_HN G95X_S_NONE;
      break;
    case ST_IMPLIED_ENDDO:
      g95x_die ("Unexpected ST_IMPLIED_ENDDO");
      break;
    case ST_IMPORT:
      p = G95X_S_IMPORT;
      break;
    case ST_INQUIRE:
      p = G95X_S_INQUIRE;
      break;
    case ST_INTERFACE:
      p = G95X_S_INTERFACE;
      break;
    case ST_PARAMETER:
      p = G95X_S_PARAMETER;
      break;
    case ST_PRIVATE:
      p = G95X_S_PRIVATE;
      break;
    case ST_PUBLIC:
      p = G95X_S_PUBLIC;
      break;
    case ST_MODULE:
      p = G95X_S_MODULE;
      break;
    case ST_MODULE_PROC:
      p = G95X_S_MODULE G95X_S_HN G95X_S_PROCEDURE;
      break;
    case ST_NAMELIST:
      p = G95X_S_NAMELIST;
      break;
    case ST_NULLIFY:
      p = G95X_S_NULLIFY;
      break;
    case ST_OPEN:
      p = G95X_S_OPEN;
      break;
    case ST_PAUSE:
      p = G95X_S_PAUSE;
      break;
    case ST_PROGRAM:
      p = G95X_S_PROGRAM;
      break;
    case ST_READ:
      p = G95X_S_READ;
      break;
    case ST_RETURN:
      p = G95X_S_RETURN;
      break;
    case ST_REWIND:
      p = G95X_S_REWIND;
      break;
    case ST_STOP:
      p = G95X_S_STOP;
      break;
    case ST_SUBROUTINE:
      p = G95X_S_SUBROUTINE;
      break;
    case ST_TYPE:
      g95x_die ("Unexpected ST_TYPE");
      break;
    case ST_USE:
      p = G95X_S_USE;
      break;
    case ST_WAIT:
      p = G95X_S_WAIT;
      break;
    case ST_WHERE_BLOCK:
      p = G95X_S_WHERE G95X_S_HN G95X_S_CONSTRUCT;
      break;
    case ST_WHERE:
      p = G95X_S_WHERE;
      break;
    case ST_WRITE:
      if (sta->code->x.is_print)
	p = G95X_S_PRINT;
      else
	p = G95X_S_WRITE;
      break;
    case ST_ASSIGN:
      p = G95X_S_ASSIGN;
      break;
    case ST_ASSIGNMENT:
      p = G95X_S_ASSIGNMENT;
      break;
    case ST_POINTER_ASSIGNMENT:
      p = G95X_S_POINTER G95X_S_HN G95X_S_ASSIGNMENT;
      break;
    case ST_SELECT_CASE:
      p = G95X_S_SELECT G95X_S_HN G95X_S_CASE;
      break;
    case ST_SEQUENCE:
      p = G95X_S_SEQUENCE;
      break;
    case ST_SIMPLE_IF:
      p = G95X_S_IF;
      break;
    case ST_STATEMENT_FUNCTION:
      p = G95X_S_STATEMENT G95X_S_HN G95X_S_FUNCTION;
      break;

    case ST_CRITICAL:
      p = G95X_S_CRITICAL;
      break;
    case ST_END_CRITICAL:
      p = G95X_S_END G95X_S_HN G95X_S_CRITICAL;
      break;
    case ST_SYNC_ALL:
      p = G95X_S_SYNC G95X_S_HN G95X_S_ALL;
      break;
    case ST_SYNC_IMAGES:
      p = G95X_S_SYNC G95X_S_HN G95X_S_IMAGES;
      break;
    case ST_SYNC_TEAM:
      p = G95X_S_SYNC G95X_S_HN G95X_S_TEAM;
      break;
    case ST_SYNC_MEMORY:
      p = G95X_S_SYNC G95X_S_HN G95X_S_MEMORY;
      break;
    case ST_ALL_STOP:
      p = G95X_S_ALL G95X_S_HN G95X_S_STOP;
      break;
    case ST_NOTIFY:
    case ST_QUERY:
    case ST_NONE:
      g95x_die ("Bad statement code for statement %R", sta);
    }

  return p;
}


const char *
g95x_interface_type (interface_type it)
{
  const char *p = NULL;

  switch (it)
    {
    case INTERFACE_NAMELESS:
      p = G95X_S_NAMELESS;
      break;
    case INTERFACE_GENERIC:
      p = G95X_S_GENERIC;
      break;
    case INTERFACE_INTRINSIC_OP:
      p = G95X_S_INTRINSIC G95X_S_HN G95X_S_OPERATOR;
      break;
    case INTERFACE_USER_OP:
      p = G95X_S_USER G95X_S_HN G95X_S_OPERATOR;
      break;
    case INTERFACE_ABSTRACT:
      p = G95X_S_ABSTRACT;
      break;
    }

  return p;
}

const char *
g95x_iop_type (g95_intrinsic_op e)
{
  const char *p = NULL;

  switch (e)
    {
    case INTRINSIC_UPLUS:
    case INTRINSIC_PLUS:
      p = G95X_S_OP_PLUS;
      break;
    case INTRINSIC_UMINUS:
    case INTRINSIC_MINUS:
      p = G95X_S_OP_MINUS;
      break;
    case INTRINSIC_TIMES:
      p = G95X_S_OP_TIMES;
      break;
    case INTRINSIC_DIVIDE:
      p = G95X_S_OP_DIVIDE;
      break;
    case INTRINSIC_POWER:
      p = G95X_S_OP_POWER;
      break;
    case INTRINSIC_CONCAT:
      p = G95X_S_OP_CONCAT;
      break;
    case INTRINSIC_AND:
      p = G95X_S_OP_AND;
      break;
    case INTRINSIC_OR:
      p = G95X_S_OP_OR;
      break;
    case INTRINSIC_EQV:
      p = G95X_S_OP_EQV;
      break;
    case INTRINSIC_NEQV:
      p = G95X_S_OP_NEQV;
      break;
    case INTRINSIC_EQ:
      p = G95X_S_OP_EQ;
      break;
    case INTRINSIC_NE:
      p = G95X_S_OP_NE;
      break;
    case INTRINSIC_GT:
      p = G95X_S_OP_GT;
      break;
    case INTRINSIC_GE:
      p = G95X_S_OP_GE;
      break;
    case INTRINSIC_LT:
      p = G95X_S_OP_LT;
      break;
    case INTRINSIC_LE:
      p = G95X_S_OP_LE;
      break;
    case INTRINSIC_NOT:
      p = G95X_S_OP_NOT;
      break;
    case INTRINSIC_ASSIGN:
      p = G95X_S_OP_ASSIGN;
      break;
    default:
      g95x_die ("Bad op code");
    }

  return p;
}

const char *
g95x_ref_type (g95_ref * r)
{
  const char *p = "";
  int i;

  switch (r->type)
    {
    case REF_ARRAY:
      {
	switch (r->u.ar.type)
	  {
	  case AR_ELEMENT:
	    p = G95X_S_ARRAY G95X_S_HN G95X_S_ELEMENT;
	    break;
	  case AR_SECTION:
	    for (i = 0; i < r->u.ar.dimen; i++)
	      if (r->u.ar.start[i] || r->u.ar.end[i] || r->u.ar.stride[i])
		return G95X_S_ARRAY G95X_S_HN G95X_S_SECTION;
	  case AR_FULL:
	    p = G95X_S_ARRAY G95X_S_HN G95X_S_FULL;
	    break;
	  case AR_UNKNOWN:
	    g95x_die ("Unknown array ref %R", r);
	    break;
	  }
      }
      break;
    case REF_COMPONENT:
      p = G95X_S_STRUCTURE G95X_S_HN G95X_S_COMPONENT;
      break;
    case REF_SUBSTRING:
      p = G95X_S_SUBSTRING;
      break;
    case REF_COARRAY:
      p = G95X_S_CO G95X_S_HN G95X_S_ARRAY G95X_S_HN G95X_S_ELEMENT;
      break;
    }
  return p;
}


const char *
g95x_flavor_type (g95_symbol * sym)
{
  symbol_attribute *a = &sym->attr;
  const char *p = NULL;

  switch (a->flavor)
    {
    case FL_PROGRAM:
      p = G95X_S_PROGRAM;
      break;
    case FL_BLOCK_DATA:
      p = G95X_S_BLOCK G95X_S_HN G95X_S_DATA;
      break;
    case FL_MODULE:
      p = G95X_S_MODULE;
      break;
    case FL_VARIABLE:
      p = G95X_S_VARIABLE;
      break;
    case FL_PARAMETER:
      p = G95X_S_PARAMETER;
      break;
    case FL_LABEL:
      p = G95X_S_LABEL;
      break;
    case FL_PROCEDURE:
      if (a->function || (sym->ts.type != BT_PROCEDURE))
	{
	  p = G95X_S_FUNCTION;
	}
      else if (a->subroutine)
	{
	  p = G95X_S_SUBROUTINE;
	}
      else
	{
	  p = G95X_S_PROCEDURE;
	}
      break;
    case FL_DERIVED:
      p = G95X_S_DERIVED;
      break;
    case FL_NAMELIST:
      p = G95X_S_NAMELIST;
      break;
    case FL_UNKNOWN:
      g95x_die ("");
      break;
    }

  return p;
}

const char *
g95x_expr_type (g95_expr * e)
{
  const char *p = NULL;

  switch (e->type)
    {
    case EXPR_OP:
      {
	if (e->value.op.operator == INTRINSIC_PAREN)
	  p = G95X_S_PAREN;
	else
	  p = G95X_S_OPERATOR;
	break;
      }
    case EXPR_FUNCTION:
      {
	if (e->x.is_operator)
	  {
	    p = G95X_S_OPERATOR;
	  }
	else
	  {
	    p = G95X_S_FUNCTION G95X_S_HN G95X_S_REFERENCE;
	  }
	break;
      }
    case EXPR_CONSTANT:
      p = G95X_S_LITERAL;
      break;
    case EXPR_VARIABLE:
      p = G95X_S_VARIABLE;
      break;
    case EXPR_SUBSTRING:
      p = G95X_S_LITERAL;
      break;
    case EXPR_STRUCTURE:
      p = G95X_S_STRUCTURE G95X_S_HN G95X_S_CONSTRUCTOR;
      break;
    case EXPR_PROCEDURE:
      p = G95X_S_PROCEDURE;
      break;
    case EXPR_ARRAY:
      p = G95X_S_ARRAY G95X_S_HN G95X_S_CONSTRUCTOR;
      break;
    case EXPR_NULL:
      p = G95X_S_FUNCTION G95X_S_HN G95X_S_REFERENCE;
      break;
    default:
      g95x_die ("Bad expr code for %R", e);
    }

  return p;
}


void
g95x_resolve_iop (g95_intrinsic_op type, g95_namespace * ns, void **p)
{
  *p = NULL;
  while (ns)
    {
      *p = &ns->operator[type];
      if (*p)
	return;
      if (ns->interface && (!ns->import))
	break;
      ns = ns->parent;
    }
  *p = (void *) g95x_iop_type (type);
}

void
g95x_resolve_uop (char *name, g95_namespace * ns, void **p)
{
  *p = NULL;
  while (ns)
    {
      *p = g95_find_symtree (ns->uop_root, name);
      if (*p)
	return;
      if (ns->interface && (!ns->import))
	break;
      ns = ns->parent;
    }
}

void
g95x_resolve_gen (char *name, g95_namespace * ns, void **p)
{
  *p = NULL;
  while (ns)
    {
      g95_symtree *st = g95_find_symtree (ns->generic_root, name);
      *p = st;
      if (*p)
	return;
      if (ns->interface && (!ns->import))
	break;
      ns = ns->parent;
    }
}

void
g95x_resolve_sym (char *name, g95_namespace * ns, void **p)
{
  g95_symbol *s;
  *p = NULL;
  while (ns)
    {
      if (g95_find_symbol (name, ns, 0, &s) == 0)
	{
	  if (s && g95x_symbol_valid (s))
	    *p = s;
	}
      if (*p)
	return;
      if (ns->interface && (!ns->import))
	break;
      ns = ns->parent;
    }
}

const char *
g95x_type_name (g95_typespec * ts)
{
  static char tmp[G95_MAX_SYMBOL_LEN+1];
  switch (ts->type)
    {
    case BT_INTEGER:
      return G95X_S_INTEGER;
    case BT_REAL:
      return ts->x.dble ? G95X_S_DOUBLE G95X_S_PRECISION : G95X_S_REAL;
    case BT_COMPLEX:
      return ts->x.dble ? G95X_S_DOUBLE G95X_S_COMPLEX : G95X_S_COMPLEX;
    case BT_LOGICAL:
      return G95X_S_LOGICAL;
      break;
    case BT_CHARACTER:
      return G95X_S_CHARACTER;
      break;
    case BT_DERIVED:
      g95x_get_name_substr (&ts->x.where_derived, tmp);
      return &tmp[0];
      break;
    case BT_PROCEDURE:
      return G95X_S_PROCEDURE;
      break;
    default:
      return NULL;
      break;
    }

  return NULL;
}

const void *
g95x_type_id (g95_typespec * ts)
{
  switch (ts->type)
    {
    case BT_INTEGER:
      return "integer";
    case BT_REAL:
      return ts->x.dble ? "doubleprecision" : "real";
    case BT_COMPLEX:
      return ts->x.dble ? "doublecomplex" : "complex";
    case BT_LOGICAL:
      return "logical";
      break;
    case BT_CHARACTER:
      return "character";
      break;
    case BT_DERIVED:
      return ts->derived;
      break;
    case BT_PROCEDURE:
      return "procedure";
      break;
    default:
      return NULL;
      break;
    }

  return NULL;
}

void
g95x_get_lk (g95_typespec * ts, g95_expr ** len, g95_expr ** kind)
{
  *len = NULL;
  *kind = NULL;

  switch (ts->type)
    {
    case BT_CHARACTER:
      if (ts->cl)
	{
	  g95_charlen *cl = ts->cl;
	  if (cl->x.alias)
	    cl = cl->x.alias;
	  *len = cl->length;
	  if (*len && (*len)->x.full)
	    *len = (*len)->x.full;
	  if (g95x_valid_expr (*len))
	    *len = *len;
	  else
	    *len = NULL;
	}
      break;
    default:
      break;
    }
  if (ts->x.kind)
    *kind = ts->x.kind->kind;
}

void
g95x_get_ac (g95_array_spec * as0, g95_coarray_spec * cas0,
	     g95_array_spec ** as1, g95_coarray_spec ** cas1)
{
  *as1 = NULL;
  *cas1 = NULL;

  if (as0)
    {
      if (as0->x.alias)
	as0 = (g95_array_spec *) as0->x.alias;
      *as1 = as0;
    }

  if (cas0)
    {
      if (cas0->x.alias)
	cas0 = (g95_coarray_spec *) cas0->x.alias;
      *cas1 = cas0;
    }

}

void
g95x_get_c_acv (g95_component * comp,
		g95_array_spec ** as, g95_coarray_spec ** cas,
		g95_expr ** val)
{
  g95_array_spec *as0 = (g95_array_spec *) comp->as;
  g95_coarray_spec *cas0 = (g95_coarray_spec *) comp->cas;

  g95x_get_ac (as0, cas0, as, cas);

  *val = NULL;

  if (comp->initializer)
    {
      g95_expr *val0 = comp->initializer;
      if (val0 && val0->x.full)
	val0 = val0->x.full;
      if (g95x_valid_expr (val0))
	*val = val0;
    }
}


void
g95x_get_s_acv (g95_symbol * sym,
		g95_array_spec ** as, g95_coarray_spec ** cas,
		g95_expr ** val)
{
  g95_array_spec *as0 = (g95_array_spec *) sym->as;
  g95_coarray_spec *cas0 = (g95_coarray_spec *) sym->cas;

  g95x_get_ac (as0, cas0, as, cas);

  *val = NULL;

  if (sym->value)
    {
      g95_expr *val0 = sym->value;
      if (val0 && val0->x.full)
	val0 = val0->x.full;
      if (g95x_valid_expr (val0))
	*val = val0;
    }
}

void
g95x_get_s_lkacv (g95_symbol * sym, g95_expr ** len, g95_expr ** kind,
		  g95_array_spec ** as, g95_coarray_spec ** cas,
		  g95_expr ** val)
{
  g95x_get_lk (&sym->ts, len, kind);
  g95x_get_s_acv (sym, as, cas, val);
}

void
g95x_get_c_lkacv (g95_component * comp, g95_expr ** len, g95_expr ** kind,
		  g95_array_spec ** as, g95_coarray_spec ** cas,
		  g95_expr ** val)
{
  g95x_get_lk (&comp->ts, len, kind);
  g95x_get_c_acv (comp, as, cas, val);
}



const char *
g95x_array_spec_type (g95_array_spec * as)
{
  const char *p = NULL;
  switch (as->type)
    {
    case AS_EXPLICIT:
      p = G95X_S_EXPLICIT;
      break;
    case AS_ASSUMED_SHAPE:
      p = G95X_S_ASSUMED G95X_S_HN G95X_S_SHAPE;
      break;
    case AS_DEFERRED:
      p = G95X_S_DEFERRED;
      break;
    case AS_ASSUMED_SIZE:
      p = G95X_S_ASSUMED G95X_S_HN G95X_S_SIZE;
      break;
    default:
      g95x_die ("Unknown shape\n");
    }
  return p;
}


const char *
g95x_coarray_spec_type (g95_coarray_spec * cas)
{
  const char *p = NULL;
  switch (cas->type)
    {
    case CAS_DEFERRED:
      p = G95X_S_DEFERRED;
      break;
    case CAS_ASSUMED:
      p = G95X_S_ASSUMED G95X_S_HN G95X_S_SIZE;
      break;
    default:
      g95x_die ("Unknown shape\n");
    }
  return p;
}




/*
 * Commons
 */

int
g95x_common_valid (g95_common_head * c)
{
  g95_symbol *sym;

  /* Don't print commons imported from another module 
     It is forbidden to mix symbols from different units
     within the same common */
  sym = c->head;

  if (sym && sym->x.imported)
    return 0;

  return 1;
}





g95x_voidp_list *
g95x_set_voidp_prev (g95x_voidp_list * f)
{
  if (f == NULL)
    return NULL;
  for (; f->next; f = f->next)
    {
      f->next->prev = f;
    }
  return f;
}





const char *
g95x_get_constant_as_string (g95_expr * h, int noboz)
{
  static char *constant = NULL;
  int length = 0;

  if (length == 0)
    {
      length = 4000;
      constant = g95_getmem (length);
    }

  if ((!noboz) && h->x.boz)
    strcpy (constant, h->x.boz);

  else

    switch (h->ts.type)
      {
      case BT_INTEGER:
	strcpy (constant, bi_to_string (h->value.integer));
	break;
      case BT_REAL:
	strcpy (constant, bg_to_string (h->value.real));
	break;
      case BT_COMPLEX:
	/* bg_to_string not re-entrant, proceed in two steps */
	constant[0] = '\0';
	strcat (constant, "(");
	strcat (constant, bg_to_string (h->value.complex.r));
	strcat (constant, ",");
	strcat (constant, bg_to_string (h->value.complex.i));
	strcat (constant, ")");
	break;
      case BT_LOGICAL:
	strcpy (constant, h->value.logical ? ".TRUE." : ".FALSE.");
	break;
      case BT_CHARACTER:
	if (length < h->value.character.length + 1)
	  {
	    g95_free (constant);
	    length = 2 * (h->value.character.length + 1);
	    constant = g95_getmem (length);
	  }
	strncpy (constant, h->value.character.string,
		 h->value.character.length);
	constant[h->value.character.length] = '\0';
	break;
      case BT_DERIVED:
	return NULL;
	break;
      case BT_UNKNOWN:
	return NULL;
	break;
      case BT_PROCEDURE:
	return NULL;
	break;
      }
  return constant;
}




const char *
g95x_formal_arg_type (g95_formal_arglist * fa)
{
  if (!fa->sym)
    return G95X_S_ALT G95X_S_HN G95X_S_RETURN;
  switch (fa->cb)
    {
    case CB_NONE:
      return G95X_S_VARIABLE;
    case CB_REFERENCE:
      return G95X_S_REFERENCE;
    case CB_VALUE:
      return G95X_S_VALUE;
    }
  return NULL;
}

#ifdef UNDEF
static char *
basename (char *filename)
{
  int i, n = strlen (filename);
  for (i = n - 1; i > 0; i--)
    {
      if (filename[i - 1] == '/')
	return &filename[i];
    }
  return filename;
}
#endif


int
g95x_start ()
{
  g95_linebuf *lb;
  int i;
  g95_file *f;

  for (lb = g95x_get_line_head (); lb; lb = lb->next)
    for (i = lb->file->x.width; i < lb->size; i++)
      lb->mask[i] = G95X_CHAR_RIMA;

  for (f = g95x_get_file_head (); f; f = f->next)
    {
      f->x.cur = f->x.text;
      f->x.linenum = 1;
    }

  g95x_add_dmp ();

  return 1;
}

int
g95x_end ()
{

  g95x_check_bbt_ids ();

  g95x_free_bbt_ids (&root_bbt_id);

  g95x_end_dmp ();

  return 1;
}
