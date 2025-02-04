def test_collector():
    from common.collector import Collector

    c = Collector()

    c("hello")
    c("goodbye")

    for i, s in enumerate(c):
        if i == 0:
            assert s == "hello"
        elif i == 1:
            assert s == "goodbye"
        else:
            assert False
