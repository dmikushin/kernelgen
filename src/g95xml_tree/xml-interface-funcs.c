

void
g95x_free_intf_info (g95x_intf_info * iif)
{
  g95x_intf_info *ii;
  while (iif)
    {
      ii = iif->next;
      g95_free (iif);
      iif = ii;
    }
}
