def check_inside_box(large_box,small_box):
    '''

    :param large_box: [left, top, right, bottom]
    :param small_box: [left, top, right, bottom]
    :return:
    '''
    left_l, top_l, right_l, bottom_l= large_box
    left_s, top_s, right_s, bottom_s = small_box
    if (left_s >= left_l) and (right_s <= right_l) and (top_s >= top_l and bottom_s <= bottom_l):
        return True
    else:
        return False
