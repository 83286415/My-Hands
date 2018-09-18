import os
import tarfile
from six.moves import urllib
import email
import email.policy
from email import parser
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import re
from html import unescape
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score


DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
WORKING_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
ROOT_PATH = os.path.join(WORKING_PATH, 'Hands on SK and TS\\')
DATA_PATH = os.path.join(ROOT_PATH, 'datasets\\')
SPAM_PATH = os.path.join(DATA_PATH, "spam")
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")


def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):  # check the spam folder exists or not
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        # for a, b in ((c, d), (e, f)): a loop in c & e while b loop in d & f respectively
        path = os.path.join(spam_path, filename)  # join the ham and spam file path
        if not os.path.isfile(path):  # check the ham or spam file exists or not
            urllib.request.urlretrieve(url, path)  # download file from website
        tar_bz2_file = tarfile.open(path)  # make dir for file downloaded and extract files from zip
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()


def load_email(is_spam, filename, spam_path=SPAM_PATH):  # is_spam: bool
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:  # open with as == try open finally close
        return email.parser.BytesParser(policy=email.policy.default).parse(f)  # parse spam or ham file in email policy
        # return a list of root structure of email (object)


def get_email_structure(email):
    if isinstance(email, str):  # blank email
        return email
    payload = email.get_payload()  # get the payload of email and return a list.

    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)  # regression in email
            for sub_email in payload
        ]))  # string like '{}, {}, {}'.format('a', 'b', 'c')  output: 'a, b, c'
    else:
        return email.get_content_type()


def structures_counter(emails):
    structures = Counter()  # define a counter to compute the counts of each structure of the emails
    for _email in emails:
        structure = get_email_structure(_email)
        structures[structure] += 1  # counts + 1, the same structure in different emails counts together
    return structures  # return an dict{structure type: counts} of all emails


def html_to_plain_text(html):   # convert html to plain text for many spam emails in html
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)  # drop head
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)  # convert <a> to hyperlink
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)                       # drop all tags
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)                 # replace multi-lines with single lines
    return unescape(text)


def email_to_text(email):
    html = None
    for part in email.walk():  # EmailMessage.walk() go through this email's each payload (iterators in email)
        ctype = part.get_content_type()  # get type
        if not ctype in ("text/plain", "text/html"):  # if not plain or html, ignore it
            continue
        try:
            content = part.get_content()
        except:  # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)  # return plain text finally


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""  # in case of no return from email to text function
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))  # set type refer to my notebook
                urls.sort(key=lambda url: len(url), reverse=True)  # sort from Z to A according to url's len
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)  # replace numbers with 'NUMBER'
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())   # a new counter from an iterable(a list here)
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()  # a new empty counter
                for word, count in word_counts.items():  # loop in text list and count the words
                    stemmed_word = stemmer.stem(word)  # stem the words
                    stemmed_word_counts[stemmed_word] += count  # count the stemmed words
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)  # append a dict{word: count} of each email to this list
        return np.array(X_transformed)


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()  # all emails counter
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)  # count the words exclude 'the' etc useless words
        most_common = total_count.most_common()[:self.vocabulary_size]  # most common used words top 1000
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        # vocabulary is a dict with most used words as key and its order number as value
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):  # enumerate make X into [(0, email_0), (1, email_1), (2, email_2), ...]
            for word, count in word_count.items():
                rows.append(row)  # X's row number as well as the number of email, 0, 1, 2 ...
                cols.append(self.vocabulary_.get(word, 0))  # get the word's index in vocabulary if none set it 0
                # get method used for dict for finding the value by key
                data.append(count)  # the count of the word most used
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
        # return a Compressed Sparse Row matrix as below:
        # csr_matrix((data, (row_ind, col_ind)), [shape = (M, N)])
        # where ``data``, ``row_ind`` and ``col_ind`` satisfy the relationship ``a[row_ind[k], col_ind[k]] = data[k]``.


if __name__ == '__main__':

    print('email going through')
    # fetch and load data
    fetch_spam_data()

    ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
    print(ham_filenames)  # file names list
    print(len(ham_filenames))  # 2500
    spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    print(ham_emails)  # return a list of email objects
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    print(spam_emails[4].get_content().strip())  # return the contents of the email
    print(spam_emails[4])  # return all contents include headers, content ...
    print(type(spam_emails[4]), 'type')  # type is email.message.EmailMessage

    print(get_email_structure(ham_emails[4]))  # show the email's structure
    print(structures_counter(ham_emails).most_common())  # show all emails' structures counts
    # if most_common(3), then show the first 3 most common counts

    for key, value in spam_emails[4].items():   # spam_email[4] is a dict
        print(key, ":", value)  # print sender, receivier, subject, etc...

    print(spam_emails[4]["Subject"])
    print('----------')

    print('data preparation')
    X = np.array(ham_emails + spam_emails)  # train data set
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))  # [0, 0, 0... 1, 1, 1... 1]  ham:0; spam:1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    html_spam_emails = [email for email in X_train[y_train == 1]  # loop all spam html emails
                        if get_email_structure(email) == "text/html"]
    sample_html_spam = html_spam_emails[7]
    print(sample_html_spam.get_content().strip()[:1000], "...")                 # html with tags
    print(html_to_plain_text(sample_html_spam.get_content()[:1000]), "...")  # plain text converted
    print(email_to_text(sample_html_spam)[:100], "...")                         # convert whole email to plain text

    try:
        import nltk  # Natural Language Toolkit

        stemmer = nltk.PorterStemmer()  # suffix stripper
        for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
            print(word, "=>", stemmer.stem(word))  # output:comput. It's a suffix stripping function.
    except ImportError:
        print("Error: stemming requires the NLTK module.")
        stemmer = None

    try:
        import urlextract  # may require an Internet connection to download root domain names

        url_extractor = urlextract.URLExtract()  # a tool which can find urls in a sentence.
        print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
    except ImportError:
        print("Error: replacing URLs requires the urlextract module.")
        url_extractor = None

    print('-----preparation done-----')

    print('make classifier')

    X_few = X_train[:3]  # train sample
    X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)  # count emails' words
    print(X_few_wordcounts)  # output: a list includes 3 counters of emails words

    vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
    # define a classifier for top 10 words of each email
    X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)  # train data
    print(X_few_vectors)  # output: (the top 10 most used words in which email, rank # in this email), used times
    print(vocab_transformer.vocabulary_)
    # output: {'the': 1, 'of': 2, 'and': 3, 'url': 4, 'to': 5, 'all': 6, 'in': 7, 'christian': 8, 'on': 9, 'by': 10}
    print('----- make classifier done -----')

    print('make pipelines and train data')

    preprocess_pipeline = Pipeline([
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer()),
    ])

    X_train_transformed = preprocess_pipeline.fit_transform(X_train)  # train data

    log_clf = LogisticRegression(random_state=42)  # define a regression
    score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
    print(score.mean())
    print('----- train data done -----')

    print('evaluate the classifier')
    X_test_transformed = preprocess_pipeline.transform(X_test)

    log_clf = LogisticRegression(random_state=42)
    log_clf.fit(X_train_transformed, y_train)

    y_pred = log_clf.predict(X_test_transformed)

    print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
    print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
    print('----- evaluation done -----')