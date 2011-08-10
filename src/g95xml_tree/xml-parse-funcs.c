
static g95_symbol *g95x_stmt_func_symbol = NULL;

static void
g95x_stmt_func_set_symbol (g95_symbol * sym)
{
  g95x_stmt_func_symbol = sym;
}

g95_symbol *
g95x_stmt_func_take_symbol ()
{
  g95_symbol *sym = g95x_stmt_func_symbol;
  g95x_stmt_func_symbol = NULL;
  return sym;
}
