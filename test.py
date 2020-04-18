from defend import Defend
from preprocessing import get_train_test_split

if __name__ == "__main__":
    platform = 'politifact'
    content = 'data/' + platform + '_content_no_ignore.tsv'
    comment = 'data/' + platform + '_comment_no_ignore.tsv'
    VALIDATION_SPLIT = 0.25

    MAX_SENTENCE_LENGTH = 120 
    MAX_COMS_LENGTH = 120

    props = get_train_test_split(content, comment, VALIDATION_SPLIT)

    id_train = props['train']['id']
    id_test = props['val']['id']
    x_train = props['train']['x']
    x_val = props['val']['x']
    y_train = props['train']['y']
    y_val  = props['val']['y']
    c_train = props['train']['c']
    c_val = props['val']['c']

    defend = Defend(platform, MAX_SENTENCE_LENGTH, MAX_COMS_LENGTH)

    defend.train(x_train, y_train, c_train, c_val, x_val, y_val, batch_size=9, epochs=5)