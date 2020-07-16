import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { MarchingViewComponent } from './marching-view.component';

describe('MarchingViewComponent', () => {
  let component: MarchingViewComponent;
  let fixture: ComponentFixture<MarchingViewComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ MarchingViewComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(MarchingViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
