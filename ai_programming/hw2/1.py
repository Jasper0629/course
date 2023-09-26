class Node:
    """节点类"""
    def __init__(self, elem, father=None, lchild=None, rchild=None):
        self.elem = elem
        self.lchild = lchild
        self.rchild = rchild
        self.father = father
    def l_add(self, elem):
        """左插"""
        node = Node(elem, self, None, None)
        if self.lchild is None:
            self.lchild = node
        else:
            print("左孩子已经存在")

    def r_add(self, elem):
        """右插"""
        node = Node(elem, self, None, None)
        if self.rchild is None:
            self.rchild = node
        else:
            print("右孩子已经存在")
class Tree:
    def __init__(self, elem) -> None:
        self.root = Node(elem,None,None,None)


    def is_empty(self):
        return self.root is None
    
    def pre_order(self, node):
        """前序遍历"""
        if node is None:
            return
        print(node.elem, end=" ")
        self.pre_order(node.lchild)
        self.pre_order(node.rchild)
    
    def pre_in_order(self, node):
        """中序遍历"""
        if node is None:
            return
        self.pre_in_order(node.lchild)
        print(node.elem, end=" ")
        self.pre_in_order(node.rchild)

    def pre_post_order(self, node):
        """后序遍历"""
        if node is None:
            return
        self.pre_post_order(node.lchild)
        self.pre_post_order(node.rchild)
        print(node.elem, end=" ")

    def level_order(self, root):
        """层次遍历，也叫广度优先搜索（Breadth-first search）。
        采用队列实现。"""
        if root is None:
            return
        queue = []
        queue.append(root)
        while queue:
            node = queue.pop(0)
            print(node.elem, end=" ")
            # 如果孩子节点不为空，那么从左至右添加进来
            if node.lchild is not None:
                queue.append(node.lchild)
            if node.rchild is not None:
                queue.append(node.rchild)

    def pre_order_stack(self, root):
        """先序遍历的非递归实现。
        使用栈"""
        if root is None:
            return
        stack = []
        node = root
        while node or stack:
            while node:
                print(node.elem, end=" ")
                stack.append(node)
                node = node.lchild
            node = stack.pop()
            node = node.rchild

    def in_order_stack(self, root):
        """中序遍历的非递归实现。
        使用栈"""
        if root is None:
            return
        stack = []
        node = root
        while node or stack:
            while node:
                stack.append(node)
                node = node.lchild
            node = stack.pop()
            print(node.elem, end=" ")
            node = node.rchild

    def post_order_stack(self, root):
        """后序遍历的非递归实现。
        使用栈"""
        if root is None:
            return
        stack1, stack2 = [], []
        stack1.append(root)
        while stack1:
            # 此循环为了找出后续遍历的逆序，存入stack2中
            # 先将当前节点拿出来存入stack2中
            node = stack1.pop()
            stack2.append(node)
            if node.lchild is not None:
                # 若左孩子非空，先入栈stack1（先入后出，所以是逆序）
                stack1.append(node.lchild)
            if node.rchild is not None:
                # 若右孩子非空，入栈stack1
                stack1.append(node.rchild)
        for i in stack2[::-1]:
            print(i.elem, end=" ")

if __name__ == "__main__":
    tree = Tree(3)
    tree.root.l_add(1)
    tree.root.r_add(2)
    tree.root.rchild.l_add(4)
    print("递归先序遍历：")
    tree.pre_order(tree.root)
    print()
    print("递归中序遍历：")
    tree.pre_in_order(tree.root)
    print()
    print("递归后序遍历：")
    tree.pre_post_order(tree.root)
    print()
    print("层次遍历：")
    tree.level_order(tree.root)
    print()
    print("非递归先序遍历：")
    tree.pre_order_stack(tree.root)
    print()
    print("非递归中序遍历：")
    tree.in_order_stack(tree.root)
    print()
    print("非递归后序遍历：")
    tree.post_order_stack(tree.root)