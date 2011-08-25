int
g95x_get_only_flag ()
{
  return only_flag;
}

static void
set_local_name_locus_sym (const char *p, const char *name, g95_symbol * sym)
{
  g95_use_rename *u;

  for (u = rename_list; u; u = u->next)
    {
      if (u->x.done)
	continue;
      if (u->operator >= 0)
	continue;
      if (((u->local_name[0] != '\0') &&
	   (strcmp (u->local_name, p) == 0))
	  || (strcmp (u->use_name, p) == 0))
	{
	  u->x.done++;
	  u->x.type = -1;
	  u->x.u_use.sym = sym;
	  u->x.u_local.sym = sym;

	  if (u->x.local_name_el)
	    {
	      u->x.local_name_el->type = G95X_LOCUS_NAME;
	      u->x.use_name_el->type = G95X_LOCUS_ASSC;
	    }
	  else
	    {
	      u->x.use_name_el->type = G95X_LOCUS_NAME;
	    }

	}
    }

  if (strcmp (p, name))
    sym->x.use_assoc = 1;


}

static void
set_local_name_locus_gen (const char *p, g95_symtree * st,
			  g95_interface * intf)
{
  g95_use_rename *u;

  for (u = rename_list; u; u = u->next)
    {
      if (u->x.done)
	continue;
      if (u->operator >= 0)
	continue;
      if (((u->local_name[0] != '\0') &&
	   (strcmp (u->local_name, p) == 0))
	  || (strcmp (u->use_name, p) == 0))
	{
	  u->x.done++;
	  u->x.type = INTERFACE_GENERIC;
	  u->x.u_local.st_gnrc = st;
	  u->x.u_use.gnrc = intf;

	  if (u->x.local_name_el)
	    {
	      u->x.local_name_el->type = G95X_LOCUS_NAME;
	      u->x.use_name_el->type = G95X_LOCUS_ASSC;
	    }
	  else
	    {
	      u->x.use_name_el->type = G95X_LOCUS_NAME;
	    }
	}
    }


}

static void
set_local_name_locus_uop (const char *p, g95_user_op * uop,
			  g95_interface * intf)
{
  g95_use_rename *u;

  for (u = rename_list; u; u = u->next)
    {
      if (u->x.done)
	continue;
      if (u->operator < 0)
	continue;
      if (((u->local_name[0] != '\0') &&
	   (strcmp (u->local_name, p) == 0))
	  || (strcmp (u->use_name, p) == 0))
	{
	  u->x.done++;
	  u->x.type = INTERFACE_USER_OP;
	  u->x.u_local.uop = uop;
	  u->x.u_use.uop = intf;

	  if (u->x.local_name_el)
	    {
	      u->x.local_name_el->type = G95X_LOCUS_USOP;
	      u->x.use_name_el->type = G95X_LOCUS_ASSC;
	    }
	  else
	    {
	      u->x.use_name_el->type = G95X_LOCUS_USOP;
	    }
	}

    }


}


g95_use_rename *
g95x_module_take_rename_list ()
{
  g95_use_rename *list = rename_list;
  rename_list = NULL;
  return list;
}

static g95x_voidp_list g95x_mod;

static void
g95x_module_set_mod (g95_symbol * sym, g95x_locus * xw)
{
  g95x_mod.u.symbol = sym;
  g95x_mod.where = *xw;
}

g95x_voidp_list
g95x_module_get_mod ()
{
  return g95x_mod;
}
