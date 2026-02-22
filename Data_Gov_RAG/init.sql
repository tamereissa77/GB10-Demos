-- Data Governance RAG Demo — PostgreSQL Seed Data
-- This file runs automatically on first container start.

CREATE TABLE IF NOT EXISTS employee_salaries (
    id SERIAL PRIMARY KEY,
    employee_name VARCHAR(100) NOT NULL,
    department VARCHAR(50) NOT NULL,
    position VARCHAR(100) NOT NULL,
    base_salary DECIMAL(12,2) NOT NULL,
    bonus DECIMAL(12,2) DEFAULT 0,
    benefits_package VARCHAR(20) NOT NULL,  -- Gold, Silver, Bronze
    hire_date DATE NOT NULL
);

INSERT INTO employee_salaries (employee_name, department, position, base_salary, bonus, benefits_package, hire_date) VALUES
    ('Ahmed Hassan',    'Engineering',  'Senior Software Engineer', 150000.00, 20000.00, 'Gold',   '2020-03-15'),
    ('Mohamed Ali',     'Engineering',  'Tech Lead',                180000.00, 30000.00, 'Gold',   '2018-06-01'),
    ('Sara Johnson',    'HR',           'HR Director',              160000.00, 25000.00, 'Gold',   '2017-01-10'),
    ('James Wilson',    'Finance',      'Financial Analyst',        120000.00, 15000.00, 'Silver', '2021-09-20'),
    ('Fatima Al-Rashid','Marketing',    'Marketing Manager',        130000.00, 18000.00, 'Silver', '2019-11-05'),
    ('Omar Khalil',     'Engineering',  'Junior Developer',         85000.00,  8000.00,  'Bronze', '2023-02-14'),
    ('Lisa Chen',       'Finance',      'CFO',                      450000.00, 75000.00, 'Gold',   '2015-04-22'),
    ('David Brown',     'Operations',   'Operations Manager',       110000.00, 12000.00, 'Silver', '2022-07-30'),
    ('Nora Ahmed',      'HR',           'Recruiter',                95000.00,  10000.00, 'Bronze', '2023-08-01'),
    ('Robert Taylor',   'Engineering',  'CEO',                      500000.00, 100000.00,'Gold',   '2014-01-01');
