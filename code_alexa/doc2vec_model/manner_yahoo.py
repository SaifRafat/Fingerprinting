import pandas as pd
import os

# Path to the Yahoo training data
csv_path = r'E:\Research work\wenggang\paper\code_alexa\doc2vec_model\toy_data\yahoo\train.csv'
output_path = r'E:\Research work\wenggang\paper\code_alexa\doc2vec_model\toy_data\yahoo\manner_train.csv'

# Read the CSV file
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

print(f"Total questions in dataset: {len(df)}")

# Filter for manner questions - any question starting with "how"
def is_manner_question(question):
    if pd.isna(question):
        return False
    question_lower = str(question).lower().strip()
    return question_lower.startswith('how')

# Apply filter
manner_df = df[df['question_title'].apply(is_manner_question)].copy()

print(f"\nManner questions found: {len(manner_df)}")
print(f"Percentage of manner questions: {len(manner_df)/len(df)*100:.2f}%")

# Save the manner questions to a new CSV file
print(f"\nSaving manner questions to {output_path}...")
manner_df.to_csv(output_path, index=False)
print(f"✓ Successfully saved {len(manner_df)} manner questions")

# Display some examples
print("\nSample manner questions:")
print("-" * 80)
for i, question in enumerate(manner_df['question_title'].head(10), 1):
    print(f"{i}. {question}")

