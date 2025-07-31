import numpy as np
import pandas as pd


class UserProfileGenerator:
    """Generate realistic user profiles with age, gender, religion, and health conditions."""
    
    # Class-level constants
    AGE_GROUPS = [
        ('infants & toddlers', 0, 3),
        ('children', 4, 12),
        ('adolescents', 13, 19),
        ('young adults', 20, 34),
        ('middle-aged', 35, 59),
        ('young elderly', 60, 74),
        ('elderly', 75, 89),
        ('long-lived elderly', 90, 100)
    ]
    
    AGE_PROBABILITIES = [0.05, 0.20, 0.20, 0.20, 0.20, 0.10, 0.04, 0.01]
    
    ATTRIBUTES = {
        'gender': (['male', 'female'], None),
        'religion': (['Christianity', 'Islam', 'Buddhism', 'None'], [0.3, 0.2, 0.2, 0.3]),
        'physical': (['visual impairment', 'hearing impairment', 'intellectual disability', 'healthy'], 
                    [0.2, 0.2, 0.2, 0.4]),
        'mental': (['depression', 'anxiety', 'war', 'sexual assault', 'major accidents', 'natural disasters', 'healthy'], 
                  [0.15, 0.15, 0.075, 0.075, 0.075, 0.075, 0.4])
    }

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def _sample_age(self):
        """Sample age and corresponding age group."""
        idx = np.random.choice(len(self.AGE_GROUPS), p=self.AGE_PROBABILITIES)
        name, low, high = self.AGE_GROUPS[idx]
        age = np.random.randint(low, high + 1)
        return age, name

    def _is_realistic(self, record: dict) -> bool:
        """Check if the generated profile is realistic based on age constraints."""
        age = record['Age']
        age_group = record['Age_Group']
        mental_condition = record['Mental_Condition']
        religion = record['Religion']
        
        # Age-based mental condition constraints
        if age_group == 'infants & toddlers' and mental_condition not in {'healthy', 'natural disasters'}:
            return False
        if age_group == 'children' and mental_condition in {'war', 'sexual assault'}:
            return False
        if mental_condition == 'war' and age < 18:
            return False
        if mental_condition == 'sexual assault' and age < 10:
            return False
        if mental_condition in {'major accidents', 'depression', 'anxiety'} and age < 5:
            return False
        if religion != 'None' and age < 8:
            return False
            
        return True

    def generate_users(self, n_users: int = 100) -> pd.DataFrame:
        """Generate n_users realistic user profiles."""
        data = []
        
        while len(data) < n_users:
            age, age_group = self._sample_age()
            
            # Generate other attributes
            gender = np.random.choice(*self.ATTRIBUTES['gender'])
            religion = np.random.choice(*self.ATTRIBUTES['religion'])
            physical = np.random.choice(*self.ATTRIBUTES['physical'])
            mental = np.random.choice(*self.ATTRIBUTES['mental'])
            
            record = {
                'Age': age,
                'Age_Group': age_group,
                'Gender': gender,
                'Religion': str(religion),
                'Physical_Condition': str(physical),
                'Mental_Condition': str(mental)
            }
            
            if self._is_realistic(record):
                data.append(record)
                
        return pd.DataFrame(data)


def main():
    """Generate user profiles and save to CSV."""
    generator = UserProfileGenerator(seed=2025)
    df = generator.generate_users(10)
    df.to_csv("users_debug.csv", index=False)
    print(df.head(10))


if __name__ == "__main__":
    main()