import { usePipelineStore } from '../../store/pipelineStore';

interface Props {
  categories: number[];
}

export function CategoryBar({ categories }: Props) {
  const { selectedCategory, setSelectedCategory } = usePipelineStore();
  
  return (
    <div className="category-bar">
      <button 
        className={`category-item ${selectedCategory === null ? 'category-item--active' : ''}`}
        onClick={() => setSelectedCategory(null)}
      >
        All Items
      </button>
      {categories.sort((a, b) => a - b).map(catId => (
        <button
          key={catId}
          className={`category-item ${selectedCategory === catId ? 'category-item--active' : ''}`}
          onClick={() => setSelectedCategory(catId)}
        >
          Category {catId}
        </button>
      ))}
    </div>
  );
}
