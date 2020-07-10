import re

feature_names = ['invalid_location_character_count']

def process_dataset(df):
    df2 = df.copy()

    _add_location_invalid_character_count_feature(df2)
    
    return df2

def _add_location_invalid_character_count_feature(df):
	invalid_characters_regex = '#|\$|\|%|\?|!|/|;|@|\+|\*|\d'
    pattern = re.compile(invalid_characters_regex)
    
    def count_invalid_chars(x):
        if type(x) is float:
            return 0
        return len(re.findall(pattern, x))
    
    df['invalid_location_character_count'] = df['location'].map(count_invalid_chars)