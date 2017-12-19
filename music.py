# Imports
import json
import pylast
import random

API_KEY = "b0356f7a92ada951f46dac5d974d1490"
API_SECRET = "3ee63469d2d1eda8a0a34845364b8af6"

# Sequence size
SIZE = 4

DATAFILE = "history_train1.json"


class Entries:
    entries = []
    tags = []
    titles = []
    title_ids = {}
    tag_ids = {}

    def __init__(self, count):
        username = input("Username: ")
        try:
            network = pylast.LastFMNetwork(api_key=API_KEY,
                                           api_secret=API_SECRET)
        except pylast.WSError:
            raise Exception("Incorrect Login")

        self.entries = network.get_user(username).get_recent_tracks(limit=count)
        print("%d entries downloaded!" % len(self.entries))

    def genTitles(self):
        print("Retrieving titles...")
        self.titles = [song.track.title for song in self.entries]
        print("Done")

    def genTags(self):
        print("Retrieving tags...")
        for song in self.entries:
            try:
                self.tags.append(song.track.get_top_tags()[0].item.name)
            except IndexError:
                self.tags.append("na")
        print("Done")

    def uniqueTitles(self):
        return list(set(self.titles))

    def uniqueTags(self):
        return list(set(self.tags))

    def genTitleIds(self):
        """Create a dict with string keys and onehot vector values."""
        print("Generating title ids...")
        uniques = self.uniqueTitles()
        self.title_ids = {title: self.pad(i, len(uniques))
                          for i, title in enumerate(uniques)}
        print("Done")

    def genTagIds(self):
        """Create a dict with string keys and onehot vector values."""
        print("Generating tag ids...")
        uniques = self.uniqueTags()
        self.tag_ids = {tag: self.pad(i, len(uniques))
                        if tag is not "na" else [0] * len(uniques)
                        for i, tag in enumerate(uniques)}
        print("Done")

    def pad(self, pos, length):
        """Create a onehot vector."""
        temp = [0] * length
        temp[pos] = 1
        return temp

    def setup(self):
        self.genTags()
        self.genTagIds()
        self.genTitles()
        self.genTitleIds()

    def sumvec(self, vec1, vec2):
        """Add two onehot vectors."""
        result = []
        for i in range(min(len(vec1), len(vec2))):
            result.append(vec1[i] + vec2[i])
        return result

    def sumrange(self, col, start, dist):
        """Fold the sumvec function over the data"""
        end = start + dist
        length = len(col)
        result = [0] * length
        if start >= length:
            return result
        for i in range(start, min(length, end)):
            result = self.sumvec(result, col[i])
        return result

    def genData(self):
        """Generate a list of input/output tuples and the title dict."""
        titles = [self.title_ids[song] for song in self.titles]
        tags = [self.tag_ids[tag] for tag in self.tags]
        train_input = [self.sumrange(titles, i, SIZE)
                       for i in range(0, len(titles) - SIZE)]
        train_labels = [self.sumrange(tags, i, SIZE)
                        for i in range(0, len(tags) - SIZE)]
        expected = titles[SIZE-1::SIZE]
        return [(train_input[i] + train_labels[i], expected[i])
                for i in range(len(expected))], self.title_ids


def main():
    train_ratio = 0.8
    DOWNL = int(input("Number of entries to download or 0 to download all: "))
    print("Downloading {} entries...".format(DOWNL if DOWNL != 0 else "all"))
    entries = Entries(DOWNL if DOWNL != 0 else None)
    print("Processing data...")
    entries.setup()
    print("Processed!\nPreparing training data...")
    data, ids = entries.genData()
    random.shuffle(data)
    ll = len(data)
    l = int(train_ratio * ll)
    print("Dumping data to %s" % DATAFILE)
    with open(DATAFILE, 'w') as file:
        json.dump({"train": data[:l],
                   "test": data[l:ll],
                   "keys": ids}, file)
    print("All done!")

if __name__ == "__main__":
    main()
