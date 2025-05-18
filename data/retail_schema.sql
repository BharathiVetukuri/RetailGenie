
-- Table: transactions
CREATE TABLE transactions (
    transaction_id INT PRIMARY KEY,
    product_id INT,
    product_name VARCHAR(100),
    category VARCHAR(50),
    quantity INT,
    total_price DECIMAL(10, 2),
    store_id INT,
    date DATE
);

-- Table: returns
CREATE TABLE returns (
    return_id INT PRIMARY KEY,
    product_id INT,
    product_name VARCHAR(100),
    category VARCHAR(50),
    store_id INT,
    date DATE
);

-- Table: monthly_sales
CREATE TABLE monthly_sales (
    product_id INT,
    product_name VARCHAR(100),
    month INT,
    sales INT
);
