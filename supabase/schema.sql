
create table users (
  id uuid primary key,
  email text
);

create table orders (
  id uuid primary key,
  user_id uuid,
  status text,
  created_at timestamp default now()
);
