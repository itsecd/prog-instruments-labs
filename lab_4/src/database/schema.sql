CREATE TABLE IF NOT EXISTS users
(
    tg_id INTEGER PRIMARY KEY,
    tasks_reversed INTEGER NOT NULL CHECK( tasks_reversed IN (0, 1) ) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tasks
(
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    task TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (tg_id)
);