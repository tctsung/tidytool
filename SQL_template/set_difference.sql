---- Purpose:
-- This query is designed to identify unique IDs from multiple tables & calculate the set differences


---- Usage:
-- Visualize the intersections by Venn diagram or Upset diagram

---- Output:
-- A table showing whether each ID is present in each set and the count of items in the intersection

---- Procedure:
-- 1. Get sets of unique IDs from each source table:
WITH set1 AS (
    SELECT id_col       -- 1 or more col you want to compare
    FROM src_tbl1   
    GROUP BY id_col
),
set2 AS (
    SELECT id_col
    FROM src_tbl2
    GROUP BY id_col
),
set3 AS (
    SELECT id_col
    FROM src_tbl3
    GROUP BY id_col
),

-- 2. Create a comprehensive list of all unique IDs present in any set
all_ids AS (
    SELECT id_col FROM set1
    UNION
    SELECT id_col FROM set2
    UNION
    SELECT id_col FROM set3
)

-- 3. Determine set memberships for each ID and count the occurrences of each combination
SELECT
    (CASE WHEN s1.id_col IS NOT NULL THEN 1 ELSE 0 END) AS is_in_set1,
    (CASE WHEN s2.id_col IS NOT NULL THEN 1 ELSE 0 END) AS is_in_set2,
    (CASE WHEN s3.id_col IS NOT NULL THEN 1 ELSE 0 END) AS is_in_set3,
    COUNT(*) AS intersection_count
FROM all_ids
LEFT JOIN set1 AS s1 
    ON all_ids.id_col = s1.id_col
LEFT JOIN set2 AS s2 
    ON all_ids.id_col = s2.id_col
LEFT JOIN set3 AS s3 
    ON all_ids.id_col = s3.id_col

-- Group by all the 'is_in_setN' boolean columns:
GROUP BY 1, 2, 3  
-- Optional: Order the results for consistency
ORDER BY 1, 2, 3
;