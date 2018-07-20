#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import sys


class Check(object):

    def __init__(self):
        pass

    def add_keyword(self, keyword):
        pass

    def get_keyword(self, keyword, offset=0):
        pass

    def replace_keyword(self, keyword, offset=0, mark='*'):
        pass


class TrieNode(object):

    def __init__(self, value=None):
        self._end = False
        self._child = dict()
        self._value = value

    def add(self, ch):
        if ch not in self._child:
            node = TrieNode(ch)
            self._child[ch] = node
            return node
        else:
            return self._child.get(ch)

    def is_end(self):
        return self._end

    def set_end(self, end):
        self._end = end

    def get_child(self, ch):
        return self._child.get(ch)

    def get_value(self):
        return self._value


class TrieCheck(Check):

    def __init__(self):
        super(TrieCheck, self).__init__()
        self._root = TrieNode('')

    def add_keyword(self, keyword):
        node = self._root

        for i in keyword:
            node = node.add(i)

        node.set_end(True)

    def get_keyword(self, text, offset=0):
        if sys.version_info.major == 2:
            # pylint: disable=E0602
            if not isinstance(text, unicode) or offset >= len(text):
                raise Exception('%s is not a unicode string' % str(text))
        else:
            if not isinstance(text, str) or offset >= len(text):
                raise Exception('%s is not a unicode string' % str(text))

        i = offset
        res = []

        for ch in text[offset:]:
            node = self._root
            index = i
            node = node.get_child(ch)
            path = []

            while node:
                path.append(text[index])
                if node.is_end():
                    res.append((i, ''.join(path)))
                    break
                if index + 1 == len(text):
                    break
                index += 1
                node = node.get_child(text[index])

            i += 1

        return res

    def replace_keyword(self, text, offset=0, mark='*'):
        if sys.version_info.major == 2:
            # pylint: disable=E0602
            if not isinstance(text, unicode) or offset >= len(text):
                raise Exception('%s is not a unicode string' % str(text))
        else:
            if not isinstance(text, str) or offset >= len(text):
                raise Exception('%s is not a unicode string' % str(text))

        i = offset
        li = list(text)
        for ch in text[offset:]:
            node = self._root
            index = i
            node = node.get_child(ch)

            while node:
                if node.is_end():
                    for m in range(i, index+1):
                        li[m] = mark
                    break
                if index + 1 == len(text):
                    break
                index += 1
                node = node.get_child(text[index])

            i += 1

        return ''.join(li)


def load_content(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    return content


def main():
    filepath = './data/keyword_test.txt'
    content = load_content(filepath)

    check = TrieCheck()
    check.add_keyword('魅族')
    check.add_keyword('OPPO')

    x = check.get_keyword(content)
    print(x)


if __name__ == '__main__':
    main()
