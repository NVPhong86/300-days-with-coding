# use SQL in: MySQL, SQL Server, MS Access, Oracle, Sybase, Informix, Postgres, and other database systems.
SELECT * FROM Customers;

# Some of the most important SQL Commands
SELECT - extracts data from a database
UPDATE - updates data in a database
DELETE - deletes data from a database
INSERT INTO - inserts new data into a database
CREATE DATABASE - creates a new database
ALTER DATABASE - modifies a database
CREATE TABLE - creates a new table
ALTER TABLE - modifies a table
DROP TABLE - deletes a table
CREATE INDEX - creates an index (search key)
DROP INDEX - deletes an index

# SQL Select Statement
# lựa chọn từ database rồi trả về 1 result-set chứa giá trị
Syntax
SELECT column1, column2, ...
FROM table_name;

SELECT CustomerName, City FROM Customers;


# SQL Select distinct statement
The SELECT DISTINCT statement is used to return only distinct (different) values.

SELECT Country FROM Customers;
SELECT DISTINCT Country FROM Customers;
SELECT COUNT(DISTINCT Country) FROM Customers;













