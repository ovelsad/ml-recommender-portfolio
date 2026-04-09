from sqlalchemy import create_engine
import pandas as pd

engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

query = """
SELECT * 
FROM final_post_sadyhov_neuro
LIMIT 1
"""

df = pd.read_sql(query, engine)

print(df.columns)
