from PIL import Image, ImageDraw
import numpy as np
import random
import cv2
import os
import json


def random_color():
    color = random.choice(
        ["f94144", "f3722c", "f8961e", "f9844a", "f9c74f", "90be6d", "43aa8b", "4d908e", "577590", "277da1"])
    return hex2int(color)


def hex2int(strhex):
    return (int(strhex[:2], base=16), int(strhex[2:4], base=16), int(strhex[4:], base=16))


def image_grid(imgs, rows, cols, empty_img):
    if len(imgs) < rows * cols:
        for _ in range(rows * cols - len(imgs)): imgs.append(empty_img)
    assert len(imgs) <= rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.connections = set()

    def __str__(self):
        return f'Node [{self.x},{self.y}] connections={len(self.connections)}'

    def __repr__(self):
        return str(self)

    def connect(self, other):
        assert isinstance(other, Node)
        self.connections.add(other)
        other.connections.add(self)

    def cdist(self, other):
        assert isinstance(other, Node)
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def sdist(self, other):
        assert isinstance(other, Node)
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def draw_connections(self, canvas, factor, radius=2):
        for dst_node in self.connections:
            color = random_color()
            shape = [
                (self.y * factor + factor // 2, self.x * factor + factor // 2),
                (dst_node.y * factor + factor // 2, dst_node.x * factor + factor // 2)
            ]
            canvas.line(shape, fill=color, width=0)
            canvas.ellipse(
                (
                    self.y * factor + factor // 2 - radius, self.x * factor + factor // 2 - radius,
                    self.y * factor + factor // 2 + radius, self.x * factor + factor // 2 + radius
                ),
                fill=color, outline=color)


class Symbol:
    def __init__(self, img_path=None, img_mat=None, idx=[], nodes=[]):
        assert (img_path is not None) ^ (img_mat is not None)
        if img_path is not None:
            self.img = Image.open(img_path)
            self.mat = np.array(self.img) < 125
        if img_mat is not None:
            self.mat = img_mat.copy()
            self.img = Image.fromarray((1 - self.mat.astype(np.uint8)) * 255)
        self.nodes = nodes
        self.idx = set(idx)

    def make_graph(self):
        self.nodes = [Node(x, y) for x, y in np.argwhere(self.mat)]
        for i in range(len(self.nodes)):
            src_node = self.nodes[i]
            for d in range(i + 1, len(self.nodes)):
                dst_node = self.nodes[d]
                if src_node.sdist(dst_node) == 1:
                    src_node.connect(dst_node)
        return self

    def endpoints(self):
        return [n for n in self.nodes if len(n.connections) < 2]

    def center_of_mass(self):
        return tuple(np.argwhere(self.mat == True).mean(0).astype(np.uint8).tolist())

    def components(self):
        totalLabels, label_ids, _, _ = cv2.connectedComponentsWithStats(self.mat.astype(np.uint8), 4, cv2.CV_32S)
        return [Symbol(img_mat=(label_ids == i), idx=self.idx) for i in range(1, totalLabels)]

    def rotate(self, angle):
        assert angle in (0, 90, 180, 270, 360)
        tmp = self.mat.copy()
        for _ in range(angle // 90):
            tmp = np.rot90(tmp)
        return Symbol(img_mat=tmp, idx=self.idx)

    def fliplr(self):
        return Symbol(img_mat=np.fliplr(self.mat), idx=self.idx)

    def flipud(self):
        return Symbol(img_mat=np.flipud(self.mat), idx=self.idx)

    def match(self, other):
        try:
            matching = cv2.matchTemplate(other.mat.astype(np.uint8), self.squeeze().mat.astype(np.uint8),
                                         cv2.TM_SQDIFF).astype(np.uint8)
            occurences = (matching == 0).sum()
        except Exception as e:
            return 0
        return occurences

    def locate(self, other):
        try:
            matching = cv2.matchTemplate(other.mat.astype(np.uint8), self.squeeze().mat.astype(np.uint8),
                                         cv2.TM_SQDIFF).astype(np.uint8)
        except Exception as e:
            return []
        locations = np.argwhere(matching == 0)
        return [other.pad(x, y) for x, y in locations]

    def __contains__(self, other):
        return self.match(other.squeeze()) > 0

    def pad(self, x, y):
        tmp = np.pad(self.mat, ((x, 0), (y, 0)), mode='constant', constant_values=(False, False))
        return Symbol(img_mat=tmp, idx=self.idx)

    def jointable(self, other):
        curr_endpoints = self.endpoints()
        other_endpoints = other.endpoints()
        for curr_ptn in curr_endpoints:
            for other_ptn in other_endpoints:
                dist = curr_ptn.sdist(other_ptn)
                if dist < 2:
                    return True
        return False

    @property
    def sum(self):
        return self.mat.sum()

    def __lt__(self, other):
        return self.sum < other.sum

    def __le__(self, other):
        return all([s1 <= s2 for s1, s2 in zip(self.mat.shape, other.mat.shape)])

    def __eq__(self, other):
        return np.array_equal(self.mat, other.mat)

    def __hash__(self):
        canvas = self.unsqueeze().mat.astype(np.uint8)
        return hash(''.join(str(i) for i in canvas.flatten()))

    def __repr__(self):
        return f'Symbol(({self.mat.shape[0]}, {self.mat.shape[1]}), sum={self.sum})'

    def __add__(self, other):
        tmp = np.logical_or(self.mat, other.mat)
        return Symbol(img_mat=tmp)

    def __sub__(self, other):
        tmp = np.logical_and(self.mat, np.logical_xor(self.mat, other.mat))
        return Symbol(img_mat=tmp)

    def __mul__(self, other):
        tmp = np.logical_and(self.mat, other.mat)
        return Symbol(img_mat=tmp)

    def squeeze(self):
        coords = cv2.findNonZero(self.mat.astype(np.uint8))
        x, y, w, h = cv2.boundingRect(coords)
        rect = self.mat[y:y + h, x:x + w]
        return Symbol(img_mat=rect, idx=self.idx)

    def unsqueeze(self):
        canvas = np.zeros((16, 16), dtype=np.bool8)
        w, h = self.mat.shape
        canvas[:w, :h] = self.mat
        return Symbol(img_mat=canvas, idx=self.idx)

    def exp_img(self, factor=20):
        width, height = self.unsqueeze().img.size
        tmp_img = self.unsqueeze().img.resize((width * factor, height * factor), Image.NEAREST).convert('RGB')

        canvas = ImageDraw.Draw(tmp_img)

        for node in self.nodes:
            node.draw_connections(canvas, factor)
        return add_margin(tmp_img, 10, 10, 10, 10, (255, 255, 255))

    def is_special(self):
        return np.all(self.mat[0] == np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.bool8))

    def show(self, factor=20):
        return self.exp_img(factor).show()

    def save(self, img_path):
        return self.img.save(img_path)

    def toJSON(self):
        data = self.__dict__.copy()
        data['mat'] = self.mat.astype(np.uint8).tolist()
        data['idx'] = list(data['idx'])
        del data['img']
        del data['nodes']
        return data

    @staticmethod
    def fromJSON(data):
        mat = np.array(data['mat'], dtype=np.bool8)
        return Symbol(img_mat=mat, idx=data['idx'])


class EmptySymbol(Symbol):
    def __init__(self):
        super().__init__(img_mat=np.zeros((16, 16), dtype=np.bool8))

    def squeeze(self):
        return self


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class TemplateSet:
    def __init__(self):
        self.data = dict()

    def add(self, sym):
        hash_sym = hash(sym)
        if hash_sym in self.data:
            self.data[hash_sym].idx.update(sym.idx)
        else:
            self.data[hash_sym] = sym

    def __iter__(self):
        return iter(self.data.values())

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # random.seed(388)
    # indices = [random.randint(1, 60000) for _ in range(50)]
    max_w, max_h = 0, 0
    src = 'imgs/unifont-14.0.04'
    characters = os.listdir(src)

    all_templates = TemplateSet()
    try:
        for i, c in enumerate(characters):
            sym_idx = int(c[:-4])
            symbol = Symbol(os.path.join(src, c), idx=[sym_idx, ])
            for component in symbol.components():
                component = component.squeeze()
                component.make_graph()
                max_connections = max([len(node.connections) for node in component.nodes])
                if max_connections > 2:
                    continue

                all_templates.add(component)
                if component.mat.sum() == 1: continue
                all_templates.add(component.rotate(90))
                all_templates.add(component.rotate(180))
                all_templates.add(component.rotate(270))
                component = component.fliplr()
                all_templates.add(component)
                all_templates.add(component.rotate(90))
                all_templates.add(component.rotate(180))
                all_templates.add(component.rotate(270))
                component = component.fliplr()
                component = component.flipud()
                all_templates.add(component)
                all_templates.add(component.rotate(90))
                all_templates.add(component.rotate(180))
                all_templates.add(component.rotate(270))
            print(
                f'  [{i:05d}/{len(characters)}] templates# {len(all_templates):05d} templates% {len(all_templates) / (i + 1):.02f}',
                end='\r')
    except KeyboardInterrupt:
        print()
        save = input('Save? [y/n]: ')
        if not save.lower().startswith('y'):
            exit()
    print('\nSaving NPY...')
    templates = np.stack([symbol.unsqueeze().mat for symbol in list(all_templates)])

    print('Saving JSON...')
    with open('json/templates.json', 'w') as f:
        json.dump([symbol.toJSON() for symbol in list(all_templates)], f)
    np.save(f'templates.npy', templates)

    # chunk_size = 10000
    # for chunk_idx, chunk in enumerate(divide_chunks(list(all_templates), chunk_size), 1):
    #     templates = np.stack([symbol.unsqueeze() for symbol in chunk])
    #     np.save(f'templates_{chunk_idx * chunk_size:05d}.npy', templates)
    #     print(f'Saved templates_{chunk_idx * chunk_size:05d}.npy')

    # np.save(f'templates_{chunk_size * 1000 + 1000:05d}.npy', templates)
    # imgs = [s.exp_img() for s in symbols]
    # image_grid(imgs, 5, 10).show()