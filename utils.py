def prepare_data(trn_X_Y, tst_X_Y, trn_point_features, tst_point_features, label_features,
                 trn_point_titles, tst_point_titles, label_titles, args,logging):
    """
    This function prepares the training and testing data based on:
    type of run -> with partial reveal 
    (adding edge between some true label and test point) or no reveal
    """
    

