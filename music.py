# Imports
import pylast

API_KEY = "b0356f7a92ada951f46dac5d974d1490"
API_SECRET = "3ee63469d2d1eda8a0a34845364b8af6"

# Sequence size
SIZE = 4

TRAIN = "history_train1.csv"
TEST = "history_test1.csv"


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
        # temp = [song.track.get_top_tags() for song in self.entries]
        # self.tags = [tag[0].item.name if len(tag) is not 0 else "na"
        #              for tag in temp]
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
        print("Generating title ids...")
        uniques = self.uniqueTitles()
        self.title_ids = {title: pow(2, i) for i, title in enumerate(uniques)}
        print("Done")

    def genTagIds(self):
        print("Generating tag ids...")
        self.tag_ids = {tag: pow(2, i) if tag is not "na" else 0
                        for i, tag in enumerate(self.tags)}
        print("Done")

    def setup(self):
        self.genTags()
        self.genTagIds()
        self.genTitles()
        self.genTitleIds()

    def genData(self):
        titles = [self.title_ids[song] for song in self.titles]
        tags = [self.tag_ids[tag] for tag in self.tags]
        train_input = [sum(titles[i:i + SIZE])
                       for i in range(0, len(titles), SIZE)]
        train_labels = [sum(tags[i:i + SIZE])
                        for i in range(0, len(tags), SIZE)]
        expected = titles[SIZE-1::SIZE]
        return (train_input, train_labels, expected)


def main():
    train_ratio = 0.8
    DOWNL = int(input("Number of entries to download or 0 to download all: "))
    print("Downloading {} entries...".format(DOWNL if DOWNL != 0 else "all"))
    entries = Entries(DOWNL if DOWNL != 0 else None)
    print("Processing data...")
    entries.setup()
    print("Processed!\nPreparing training data...")
    data = entries.genData()
    ll = len(data[2])
    l = int(train_ratio * ll)
    dump = [(data[0][i], data[1][i], data[2][i]) for i in range(l)]
    print("Dumping training data to %s" % TRAIN)
    with open(TRAIN, 'w') as file:
        file.write("%d,2,input,label,output\n" % len(dump))
        file.write("\n".join(["%d,%d,%d" % entry for entry in dump]))
    print("Preparing training data...")
    dump = [(data[0][i], data[1][i], data[2][i]) for i in range(l, ll)]
    print("Dumping test data to %s" % TEST)
    with open(TEST, 'w') as file:
        file.write("%d,2,input,label,output\n" % len(dump))
        file.write("\n".join(["%d,%d,%d" % entry for entry in dump]))
    print("All done!")

if __name__ == "__main__":
    main()
