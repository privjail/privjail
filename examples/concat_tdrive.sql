COPY
(
  SELECT
  CAST(column0 AS INTEGER) AS taxi_id,
  CAST(column1 AS TIMESTAMP) AS timestamp,
  CAST(column2 AS DOUBLE) AS longitude,
  CAST(column3 AS DOUBLE) AS latitude
  FROM read_csv_auto('release/taxi_log_2008_by_id/*.txt', delim=',', header=False)
)
TO 'tdrive.csv'
;
