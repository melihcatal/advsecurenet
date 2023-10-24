from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # split remaining data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_size/(1-test_size), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
