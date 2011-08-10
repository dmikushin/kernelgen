int g95x_get_only_flag() {
  return only_flag;
}

void set_local_name_locus_sym( 
  const char* p, const char* name, 
  g95_symbol* sym 
) {
  g95_use_rename *u;
    
  for( u = rename_list; u; u = u->next ) {
  
    if( u->mark_el )
      continue;
      
    if( ( strcmp( u->use_name, name ) == 0 )
     && ( ( ( u->local_name[0] == '\0' ) && ( strcmp( name, p ) == 0 ) )
       || ( ( u->local_name[0] != '\0' ) && ( strcmp( u->local_name, p ) == 0 ) )
        )
    ) {
      u->mark_el = 1;
      if( u->local_name_el ) {
      
        u->local_name_el->type = G95X_LOCUS_SMBL;
        u->local_name_el->u.smbl.symbol = sym;
        
        u->use_name_el->type = G95X_LOCUS_ASSC;
        u->use_name_el->u.assc.symbol = sym;
        
      } else {
        
        u->use_name_el->type = G95X_LOCUS_SMBL;
        u->use_name_el->u.smbl.symbol = sym;
        
      }
      
      break;
    }
  }
  
  if( strcmp( p, name ) )
    sym->x.use_assoc = 1;
  
  
}

void set_local_name_locus_gen( 
  const char* p, const char* name, 
  g95_interface* intf
) {
  g95_use_rename *u;
    
  for( u = rename_list; u; u = u->next ) {
  
    if( u->mark_el )
      continue;
      
    if( ( strcmp( u->use_name, name ) == 0 )
     && ( ( ( u->local_name[0] == '\0' ) && ( strcmp( name, p ) == 0 ) )
       || ( ( u->local_name[0] != '\0' ) && ( strcmp( u->local_name, p ) == 0 ) )
        )
    ) {
      u->mark_el = 1;
      if( u->local_name_el ) {
      
        u->local_name_el->type = G95X_LOCUS_GNRC;
        
        u->use_name_el->type = G95X_LOCUS_ASSC;
        u->use_name_el->u.assc.intf = intf;
        
      } else {
        
        u->use_name_el->type = G95X_LOCUS_GNRC;
        u->use_name_el->u.gnrc.intf = intf;
        
      }
      
      break;
    }
  }
  
  
}

void set_local_name_locus_uop( 
  const char* p, const char* name, 
  g95_interface* intf
) {
  g95_use_rename *u;
    
  for( u = rename_list; u; u = u->next ) {
  
    if( u->mark_el )
      continue;
      
    if( ( strcmp( u->use_name, name ) == 0 )
     && ( ( ( u->local_name[0] == '\0' ) && ( strcmp( name, p ) == 0 ) )
       || ( ( u->local_name[0] != '\0' ) && ( strcmp( u->local_name, p ) == 0 ) )
        )
    ) {
      u->mark_el = 1;
      if( u->local_name_el ) {
      
        u->local_name_el->type = G95X_LOCUS_USOP;
        
        u->use_name_el->type = G95X_LOCUS_ASSC;
        u->use_name_el->u.assc.intf = intf;
        
      } else {
        
        u->use_name_el->type = G95X_LOCUS_USOP;
        u->use_name_el->u.usop.intf = intf;
        
      }
      
      break;
    }
  }
  
  
}
