#include "g95.h"
#include "xml-dump-decl.h"

#define loc1( xwa, xwb ) \
  do {                               \
    (xwa)->lb1 = (xwb)->lb1;         \
    (xwa)->column1 = (xwb)->column1; \
  } while (0) 
#define loc2( xwa, xwb ) \
  do {                               \
    (xwa)->lb2 = (xwb)->lb2;         \
    (xwa)->column2 = (xwb)->column2; \
  } while (0) 
#define locv( xw )                   \
  ( (xw)->lb1 != NULL )

g95x_locus* g95x_get_data_statement_set_locus( g95_data* d ) {
  g95_data_value* val;

  if (locv(&d->x.where))
    goto done;

  for (val = d->value; val->next; val = val->next);

  loc1 (&d->x.where, g95x_get_data_var_locus (d->var));
  loc2 (&d->x.where, g95x_get_data_val_locus (val));

done:
  return &d->x.where;
}

g95x_locus* g95x_get_data_var_locus( g95_data_variable* var ) {
  if (locv(&var->x.where))
    goto done;

  if (var->expr)
    {
      var->x.where = var->expr->x.where;
    }
  else if (var->list)
    {
      g95x_locus xw1;
      g95_data_variable* v = var->list; 
      if (v->expr)
        {
          loc1 (&xw1, &v->expr->x.where);
        }
      else
        {
          g95x_locus* xw = g95x_get_data_var_locus (v);
          loc1 (&xw1, xw);
        }
      g95x_locus* xw = g95x_get_iterator_locus (&var->iter);
      loc2 (&xw1, xw);
      g95x_opl (&xw1, &var->x.where, "(", ")");
    }

done:
  return &var->x.where;
}

g95x_locus* g95x_get_data_val_locus( g95_data_value* val ) {
  g95_expr *v = val->expr, 
           *r = val->x.repeat;

  G95X_FULLIFY (v);
  G95X_FULLIFY (r);
  
  loc1( &val->x.where, &v->x.where );
  loc2( &val->x.where, &v->x.where );

  if (r)
    loc1( &val->x.where, &r->x.where );


  return &val->x.where;
}

g95x_locus* g95x_get_iterator_locus( g95_iterator* it ) {
  g95_expr *var, *start, *end, *step;

  if (! locv (&it->x.where))
    {

      var = it->var;
      start = it->start;
      end = it->end;
      step = it->step;
     
      G95X_FULLIFY (var);
      G95X_FULLIFY (start);
      G95X_FULLIFY (end);
      G95X_FULLIFY (step);
     
      loc1 (&it->x.where, &var->x.where);
      if (start)
        loc2 (&it->x.where, &start->x.where);
      if (end)
        loc2 (&it->x.where, &end->x.where);
      if (g95x_valid_expr (step))
        loc2 (&it->x.where, &step->x.where);
     
    }

  return &it->x.where;
}

g95x_locus* g95x_get_common_block_decl_locus( g95x_voidp_list* sla ) 
{
  g95x_voidp_list *sl1;
  g95x_locus *xw = &sla->where_all;
  g95x_set_voidp_prev (sla);

  if (locv(xw))
    goto done;

  if (sla->u.common != (struct g95_common_head *) g95x_dump_ns->blank_common)
      g95x_opl (&sla->where, xw, "/", "/");
  else
      loc1 (xw, &sla->prev->where);

  for (sl1 = sla->prev; sl1; sl1 = sl1->prev)
    {
      if (sl1->type != G95X_VOIDP_SYMBOL)
	break;
      loc2 (xw, &sl1->where);
    }

done:
  return xw;
}




g95x_locus* g95x_get_equivalence_set_locus( g95_equiv* f ) 
{
  g95_equiv* e;
  g95_expr* g;
  g95x_locus *xw = &f->x.where_equiv;

  if (locv(xw))
    goto done;
  
  g = f->expr;
  G95X_FULLIFY (g);
  loc1 (xw, &g->x.where);

  for (e = f; e->eq; e = e->eq);

  g = e->expr;
  G95X_FULLIFY (g);
  loc2 (xw, &g->x.where);

done:
  return xw;
}


g95x_locus* g95x_get_namelist_group_locus (g95x_voidp_list* sla)  
{
  g95x_voidp_list *sl1;
  g95x_locus *xw = &sla->where_all;
  g95x_set_voidp_prev (sla);

  if (locv(xw))
    goto done;

  g95x_opl (&sla->where, xw, "/", "/");

  for (sl1 = sla->prev; sl1; sl1 = sl1->prev)
    {
      if (sl1->type != G95X_VOIDP_SYMBOL)
	break;
      loc2 (xw, &sl1->where);
    }

done:
  return xw;
}

g95x_locus* g95x_get_forall_iterator_locus (g95_forall_iterator* fa)  
{
  g95x_locus *xw = &fa->x.where;

  if (locv(xw))
    goto done;

  loc1 (xw, &fa->var->x.where);

  if (g95x_valid_expr (fa->stride))
    loc2 (xw, &fa->stride->x.where);
  else
    loc2 (xw, &fa->end->x.where);

done:
  return xw;
}


g95x_locus* g95x_get_constructor_locus( g95_constructor* c )
{
  g95x_locus *xw = &c->x.where;

  if (locv(xw))
    goto done;

  *xw = c->expr->x.where;

  if (c->iterator)
  {
    g95x_locus* xwl = g95x_get_iterator_locus (c->iterator);
    loc2 (xw, xwl);
  }


done:
  return xw;
}


